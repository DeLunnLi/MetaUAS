#!/usr/bin/env python3
"""COCO 图像 + LaMa 多卡填充（在线掩码）预处理脚本。"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import hashlib
import io
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm


def _is_tqdm_disabled() -> bool:
    return os.environ.get("PIPELINE_DISABLE_TQDM", "0") == "1"


def _is_png_warning_quiet() -> bool:
    return os.environ.get("PIPELINE_QUIET_PNG_WARNINGS", "0") == "1"


def _tqdm(*args, **kwargs):
    kwargs.setdefault("disable", _is_tqdm_disabled())
    return tqdm(*args, **kwargs)


@contextlib.contextmanager
def _suppress_stderr(enabled: bool):
    if not enabled:
        yield
        return
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(old_stderr_fd, 2)
        os.close(old_stderr_fd)
        os.close(devnull_fd)


def _list_image_files(image_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in exts])


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _parse_cyws_pair_stem(pair_stem: str) -> Optional[Tuple[int, int]]:
    """
    解析变体 stem，返回 (idx, k)。解析失败返回 None。
    - CYWS 导出约定：``{idx}_mask{k}``
    - LaMa 中间文件为避免和其默认 glob 规则冲突，内部也可能使用 ``{idx}_v{k}``
    约定 idx 为十进制整数（COCO 文件名通常为 12 位数字）。
    """
    sep = None
    if "_mask" in pair_stem:
        sep = "_mask"
    elif "_v" in pair_stem:
        sep = "_v"
    else:
        return None

    a, b = pair_stem.rsplit(sep, 1)
    if not a.isdigit() or not b.isdigit():
        return None
    return int(a), int(b)


def _export_cyws_pair(
    *,
    cyws_root: Path,
    idx: int,
    mask_id: int,
    orig_bgr: np.ndarray,
    pred_bgr: np.ndarray,
    mask_u8: np.ndarray,
) -> None:
    """写出 CYWS 格式：images_and_masks/{idx}.png、images_and_masks/{idx}_mask{k}.png、inpainted/{idx}_mask{k}.png"""
    img_dir = cyws_root / "images_and_masks"
    inp_dir = cyws_root / "inpainted"
    _safe_mkdir(img_dir)
    _safe_mkdir(inp_dir)

    # 原图仅需写一次；存在则跳过（避免多 variant 重复写）
    orig_path = img_dir / f"{idx}.png"
    if not orig_path.exists():
        tmp_path = img_dir / f".{idx}.tmp.png"
        cv2.imwrite(str(tmp_path), orig_bgr)
        os.replace(tmp_path, orig_path)

    # mask 与 inpaint 结果
    stem = f"{idx}_mask{mask_id}"
    mask_path = img_dir / f"{stem}.png"
    pred_path = inp_dir / f"{stem}.png"
    cv2.imwrite(str(mask_path), np.where(mask_u8 > 127, 255, 0).astype(np.uint8))
    cv2.imwrite(str(pred_path), pred_bgr)


def _mask_to_bool(mask_u8: np.ndarray) -> np.ndarray:
    """统一成 bool mask（H,W）。"""
    if mask_u8.ndim == 3:
        mask_u8 = mask_u8[..., 0]
    return (mask_u8.astype(np.uint8) > 127)


def _mask_hash(mask_bool_hw: np.ndarray) -> str:
    """像素级去重用：对 mask bytes 做哈希。"""
    # np.packbits 可显著减少 bytes，但对齐/shape 变化会影响；这里直接 bytes 更直观
    b = mask_bool_hw.tobytes()
    return hashlib.sha1(b).hexdigest()


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU(a,b)，a/b 为 bool(H,W)。"""
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    if union <= 0.0:
        # 两者都为空（全 0）则视作完全相同
        return 1.0 if inter <= 0.0 else 0.0
    return inter / union


def _cyws_base_stem_of(stem: str) -> str:
    """把 pair_stem 归一到 base stem（'123_mask4'/'123_v4' -> '123'；'123' -> '123'）。"""
    if "_mask" in stem:
        return stem.split("_mask", 1)[0]
    if "_v" in stem:
        return stem.split("_v", 1)[0]
    return stem


def _sample_unique_mask(
    *,
    coco: COCO,
    anns: List[dict],
    h: int,
    w: int,
    cfg: _RunConfig,
    seen_hash_to_mask: Dict[str, np.ndarray],
) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    """
    为同一 base 图像采样一个“足够不同”的 mask。
    - seen_hash_to_mask: {hash -> bool_mask}，用于去重/相似度判断
    返回 (mask_u8, mask_bool, hash)；失败返回 None。
    """
    tries = max(1, int(cfg.cyws_unique_max_tries))
    iou_th = float(cfg.cyws_unique_iou_thresh)
    for _ in range(tries):
        msk = _sample_mask_with_ratio_limit(
            coco=coco,
            anns=anns,
            h=h,
            w=w,
            max_mask_area_ratio=cfg.max_mask_area_ratio,
            mask_resample_attempts=1,
            min_coco_ann_area=cfg.min_coco_ann_area,
        )
        if _mask_area_ratio(msk, h, w) > float(cfg.max_mask_area_ratio):
            continue
        mb = _mask_to_bool(msk)
        hsh = _mask_hash(mb)
        if hsh in seen_hash_to_mask:
            continue
        if iou_th < 1.0 and seen_hash_to_mask:
            too_similar = False
            for prev in seen_hash_to_mask.values():
                if _mask_iou(mb, prev) >= iou_th:
                    too_similar = True
                    break
            if too_similar:
                continue
        return msk, mb, hsh
    return None


@dataclass
class _UniqueState:
    # base_id(str) -> {hash -> bool_mask}
    base_seen: Dict[str, Dict[str, np.ndarray]]
    # variant_stem -> 当前 hash
    variant_hash: Dict[str, str]
    # variant_stem -> 当前 bool mask（失败时可恢复）
    variant_bool: Dict[str, np.ndarray]


def _prepare_one_input(
    *,
    pair_stem: str,
    image_path: Path,
    out_dir: Path,
    coco: COCO,
    cfg: _RunConfig,
    ustate: _UniqueState,
    remove_old_before_sample: bool,
) -> Optional[Path]:
    """
    统一“生成 mask + 写 LaMa 输入”的逻辑（首轮与重试共用）。
    返回写出的 mask 路径；失败返回 None。
    """
    with _suppress_stderr(_is_png_warning_quiet()):
        img = cv2.imread(str(image_path))
    if img is None:
        return None
    h, w = img.shape[:2]

    base_stem = _cyws_base_stem_of(pair_stem)
    if not base_stem.isdigit():
        return None
    img_id = int(base_stem)
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    msk_u8: np.ndarray | None = None
    unique_enabled = cfg.cyws_variants_per_image > 1
    if not unique_enabled:
        msk_u8 = _sample_mask_with_ratio_limit(
            coco=coco,
            anns=anns,
            h=h,
            w=w,
            max_mask_area_ratio=cfg.max_mask_area_ratio,
            mask_resample_attempts=cfg.mask_resample_attempts,
            min_coco_ann_area=cfg.min_coco_ann_area,
        )
    else:
        ustate.base_seen.setdefault(base_stem, {})
        old_h = ustate.variant_hash.get(pair_stem)
        old_b = ustate.variant_bool.get(pair_stem)
        if remove_old_before_sample and old_h is not None:
            ustate.base_seen[base_stem].pop(old_h, None)

        sampled = _sample_unique_mask(
            coco=coco,
            anns=anns,
            h=h,
            w=w,
            cfg=cfg,
            seen_hash_to_mask=ustate.base_seen[base_stem],
        )
        if sampled is None:
            if remove_old_before_sample and old_h is not None and old_b is not None:
                ustate.base_seen[base_stem][old_h] = old_b
            return None
        msk_u8, mb, hsh = sampled
        ustate.base_seen[base_stem][hsh] = mb
        ustate.variant_hash[pair_stem] = hsh
        ustate.variant_bool[pair_stem] = mb

    cv2.imwrite(str(out_dir / f"{pair_stem}.png"), img)
    mask_path = out_dir / f"{pair_stem}_mask.png"
    cv2.imwrite(str(mask_path), msk_u8)
    return mask_path


def _delete_inpaint_outputs(output_dir: Path) -> int:
    """删除已生成的 LaMa 修复图（output_dir 下的 .jpg/.jpeg/.png）。"""
    n_img = 0
    if output_dir.is_dir():
        for p in list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.jpeg")) + list(output_dir.glob("*.png")):
            if p.is_file():
                p.unlink()
                n_img += 1
    return n_img


def _slice_by_shard(items: List[Path], shard_id: int, num_shards: int) -> List[Path]:
    if num_shards <= 1:
        return items
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"Invalid shard: shard_id={shard_id}, num_shards={num_shards}")
    return [p for idx, p in enumerate(items) if idx % num_shards == shard_id]


def _load_stem_allowlist_file(path: Path) -> Set[str]:
    """每行一个 stem（可含注释行 #）；用于限制 LaMa 只处理列表内图像。"""
    out: Set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.add(line.split()[0])
    return out


def _build_mask_from_anns(coco: COCO, anns: List[dict], h: int, w: int) -> np.ndarray:
    binary_mask = np.zeros((h, w), dtype=np.uint8)
    for ann in anns:
        if not ann.get("segmentation"):
            continue
        temp_mask = coco.annToMask(ann).astype(np.uint8)
        binary_mask[temp_mask == 1] = 255
    return binary_mask


def _mask_area_ratio(mask_u8: np.ndarray, h: int, w: int) -> float:
    return float((mask_u8 > 127).sum()) / float(h * w)


def _greedy_mask_under_ratio(
    coco: COCO,
    valid_anns: List[dict],
    h: int,
    w: int,
    max_area_ratio: float,
) -> np.ndarray:
    """随机顺序贪心并集，保证前景像素数 ≤ max_area_ratio * H * W（与训练侧实例交换面积上限一致）。"""
    if not valid_anns:
        return np.zeros((h, w), dtype=np.uint8)
    max_pixels = float(h * w) * float(max_area_ratio)
    order = list(range(len(valid_anns)))
    random.shuffle(order)
    chosen: List[dict] = []
    for i in order:
        trial = chosen + [valid_anns[i]]
        mask_try = _build_mask_from_anns(coco, trial, h, w)
        if float((mask_try > 127).sum()) <= max_pixels:
            chosen.append(valid_anns[i])
    if not chosen:
        return np.zeros((h, w), dtype=np.uint8)
    return _build_mask_from_anns(coco, chosen, h, w)


def _valid_anns_for_inpaint(
    anns: List[dict],
    *,
    min_coco_ann_area: int,
) -> List[dict]:
    """论文侧更关心「清晰物体级区域」：去掉 crowd 与过小 ``area`` 的标注。"""
    out: List[dict] = []
    for a in anns:
        if not a.get("segmentation"):
            continue
        if min_coco_ann_area > 0 and int(a.get("area", 0)) < int(min_coco_ann_area):
            continue
        out.append(a)
    return out


def _sample_mask_with_ratio_limit(
    coco: COCO,
    anns: List[dict],
    h: int,
    w: int,
    max_mask_area_ratio: float,
    mask_resample_attempts: int,
    *,
    min_coco_ann_area: int = 0,
) -> np.ndarray:
    """掩码采样固定为 random_k：随机 1～K 个实例并集，面积受 ``max_mask_area_ratio`` 限制。"""
    valid_anns = _valid_anns_for_inpaint(anns, min_coco_ann_area=min_coco_ann_area)
    if not valid_anns:
        return np.zeros((h, w), dtype=np.uint8)

    max_ratio = float(max_mask_area_ratio)

    best_mask: np.ndarray | None = None
    best_ratio = 1.0
    for _ in range(mask_resample_attempts):
        select_num = random.randint(1, len(valid_anns))
        selected_anns_try = random.sample(valid_anns, select_num)
        mask_try = _build_mask_from_anns(coco, selected_anns_try, h, w)
        ratio_try = _mask_area_ratio(mask_try, h, w)
        if ratio_try < best_ratio:
            best_ratio = ratio_try
            best_mask = mask_try
        if ratio_try <= max_ratio:
            return mask_try
    if best_mask is not None and best_ratio <= max_ratio:
        return best_mask
    return _greedy_mask_under_ratio(coco, valid_anns, h, w, max_ratio)


_LAMA_INTERMEDIATE_GLOBS = (
    ".lama_inputs_*",
    ".lama_outputs_*",
    ".lama_best_*",
    ".lama_retry_in_*",
    ".lama_retry_out_*",
)


def clean_intermediate_directories(*roots: Path) -> int:
    """
    删除 LaMa 批处理遗留的临时目录（中断或异常时可能残留）。
    在多个根目录下查找，路径去重；不删除 inpainted_images / change_pairs。
    """
    uniq_roots: List[Path] = []
    seen_root: set[str] = set()
    for root in roots:
        try:
            r = root.resolve()
        except OSError:
            continue
        key = str(r)
        if key in seen_root or not r.is_dir():
            continue
        seen_root.add(key)
        uniq_roots.append(r)

    by_key: Dict[str, Path] = {}
    for r in uniq_roots:
        for pattern in _LAMA_INTERMEDIATE_GLOBS:
            for p in r.glob(pattern):
                if p.is_dir():
                    by_key[str(p.resolve())] = p
    removed_dirs = list(by_key.values())

    for p in removed_dirs:
        shutil.rmtree(p)
        print(f"[clean] 已删除 {p}", flush=True)
    return len(removed_dirs)


def _count_inpainted_jpgs(output_dir: Path) -> int:
    if not output_dir.exists():
        return 0
    return sum(1 for p in output_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"})


def _warn_inpaint_total_pbar_if_residual(output_dir: Path, total_images: int) -> None:
    """父进程 tqdm 以 output_dir 内 JPG 数量为 n；若目录非空，条会从「已有基数」爬升，易误判为本轮已跑满。"""
    n = _count_inpainted_jpgs(output_dir)
    if n <= 0:
        return
    print(
        f"[WARN] {output_dir} 内已有 {n} 张修复图（.jpg）；"
        f"总进度条统计的是该目录 JPG 数（上限 {total_images}），会从当前基数上升，不代表本轮从零开始。"
        " 若需要单次可读的进度，请在本轮前加 --reset-inpaint-data 或手动清空该目录。",
        flush=True,
    )


def _ensure_empty_dir(path: Path) -> None:
    """删除并重建目录（若不存在则创建）。"""
    if path.exists():
        shutil.rmtree(path)
    _safe_mkdir(path)


@dataclass(frozen=True)
class _RunConfig:
    """单个 inpainting shard 的所有配置参数。"""

    ann_file: Path
    image_dir: Path
    output_dir: Path

    shard_id: int
    num_shards: int
    seed: int

    lama_repo_dir: Path
    lama_model_path: Path
    lama_checkpoint: str
    lama_python: str
    chunk_size: int

    max_mask_area_ratio: float
    mask_resample_attempts: int
    min_coco_ann_area: int

    cyws_root: Optional[Path] = None
    cyws_variants_per_image: int = 1
    cyws_unique_iou_thresh: float = 0.95
    cyws_unique_max_tries: int = 50

    quiet_unique_warnings: bool = False
    stem_allowlist: Optional[Set[str]] = None


@dataclass
class _ExportCfg:
    shard_id: int
    cyws_root: Optional[Path]


def _finalize_one_result(
    *,
    stem: str,
    image_path: Path,
    pred_bgr: np.ndarray,
    mask_u8: np.ndarray,
    output_dir: Path,
    exp: _ExportCfg,
) -> None:
    """
    统一收敛“落盘与导出”逻辑：
    - 写 inpainted_images/{stem}.jpg
    - 可选导出 CYWS(coco-inpainted) 格式
    """
    _write_inpaint_jpg(output_dir, stem, pred_bgr)
    if exp.cyws_root is not None:
        parsed = _parse_cyws_pair_stem(stem)
        if parsed is not None:
            idx, mid = parsed
            with _suppress_stderr(_is_png_warning_quiet()):
                orig_bgr = cv2.imread(str(image_path))
            if orig_bgr is not None:
                _export_cyws_pair(
                    cyws_root=Path(exp.cyws_root),
                    idx=idx,
                    mask_id=mid,
                    orig_bgr=orig_bgr,
                    pred_bgr=pred_bgr,
                    mask_u8=mask_u8,
                )


def _write_inpaint_jpg(output_dir: Path, stem: str, pred_bgr: np.ndarray) -> None:
    cv2.imwrite(str(output_dir / f"{stem}.jpg"), pred_bgr)


def _run_lama_batch(
    predict_script: Path,
    lama_repo_dir: Path,
    lama_model_path: Path,
    lama_checkpoint: str,
    work_dir: Path,
    pred_out_dir: Path,
    env: dict,
    pbar_desc: str,
    lama_python: str,
) -> Tuple[int, str, List[Path]]:
    cmd = [
        lama_python,
        str(predict_script),
        f"model.path={lama_model_path}",
        f"model.checkpoint={lama_checkpoint}",
        f"indir={work_dir}",
        f"outdir={pred_out_dir}",
        "dataset.img_suffix=.png",
    ]
    stdout_log = pred_out_dir / "predict_stdout.log"
    _safe_mkdir(pred_out_dir)
    proc = subprocess.Popen(
        cmd,
        shell=False,
        cwd=str(lama_repo_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    total_inputs = sum(1 for _ in work_dir.glob("*_mask.png"))
    pbar = _tqdm(total=total_inputs, desc=pbar_desc)
    last_done = 0
    log_lines: List[str] = []

    with stdout_log.open("w", encoding="utf-8") as out_f:
        if proc.stdout is not None:
            for line in proc.stdout:
                out_f.write(line)
                out_f.flush()
                log_lines.append(line)
                if len(log_lines) > 40:
                    log_lines.pop(0)

                done = sum(1 for _ in pred_out_dir.glob("*_mask.png"))
                if done > last_done:
                    pbar.update(done - last_done)
                    last_done = done

    proc.wait()
    done = sum(1 for _ in pred_out_dir.glob("*_mask.png"))
    if done > last_done:
        pbar.update(done - last_done)
    pbar.close()

    pred_files = list(pred_out_dir.glob("*_mask.png"))
    tail = "".join(log_lines)
    return int(proc.returncode), tail, pred_files


def _prepare_masks_for_chunk(
    *,
    chunk_image_files: List[Path],
    work_dir: Path,
    coco: COCO,
    cfg: _RunConfig,
    chunk_idx: int,
    num_chunks: int,
) -> Tuple[List[Tuple[str, Path]], Dict[str, Path]]:
    """Phase 1: 为 chunk 内每一张图采样掩码，写出 LaMa 输入文件。

    Returns:
        valid_items: [(pair_stem, image_path), ...]
        current_mask_paths: {pair_stem -> mask 临时文件路径}
    """
    valid_items: List[Tuple[str, Path]] = []
    current_mask_paths: Dict[str, Path] = {}
    ustate = _UniqueState(base_seen={}, variant_hash={}, variant_bool={})

    for image_path in _tqdm(
        chunk_image_files,
        desc=f"Shard {cfg.shard_id}/{cfg.num_shards} | 准备掩码 {chunk_idx + 1}/{num_chunks}",
    ):
        stem = image_path.stem
        with _suppress_stderr(_is_png_warning_quiet()):
            img = cv2.imread(str(image_path))
        if img is None:
            continue

        try:
            int(stem)
        except ValueError:
            print(f"[WARN] Skip non-numeric stem (need COCO id as filename): {image_path.name}")
            continue

        nvar = max(1, int(cfg.cyws_variants_per_image))
        for mask_id in range(1, nvar + 1):
            pair_stem = f"{stem}_v{mask_id}" if nvar > 1 else stem
            mp = _prepare_one_input(
                pair_stem=pair_stem,
                image_path=image_path,
                out_dir=work_dir,
                coco=coco,
                cfg=cfg,
                ustate=ustate,
                remove_old_before_sample=False,
            )
            if mp is None:
                if not cfg.quiet_unique_warnings:
                    print(
                        f"[WARN] shard={cfg.shard_id} stem={stem} mask_id={mask_id}: "
                        f"无法在 {int(cfg.cyws_unique_max_tries)} 次内采到 unique mask（跳过该变体）",
                        flush=True,
                    )
                if mask_id == 1:
                    continue
                break
            valid_items.append((pair_stem, image_path))
            current_mask_paths[pair_stem] = mp

    return valid_items, current_mask_paths


def _execute_lama_chunk(
    *,
    cfg: _RunConfig,
    work_dir: Path,
    pred_out_dir: Path,
    env: dict,
    chunk_idx: int,
    num_chunks: int,
) -> Tuple[int, str, List[Path]]:
    """Phase 2: 运行 LaMa 批处理推理。"""
    predict_script = cfg.lama_repo_dir / "bin" / "predict.py"
    pbar_desc = f"Shard {cfg.shard_id}/{cfg.num_shards} | LaMa {chunk_idx + 1}/{num_chunks}"
    return _run_lama_batch(
        predict_script,
        cfg.lama_repo_dir,
        cfg.lama_model_path,
        cfg.lama_checkpoint,
        work_dir,
        pred_out_dir,
        env,
        pbar_desc,
        cfg.lama_python,
    )


def _collect_chunk_results(
    *,
    pred_files: List[Path],
    current_mask_paths: Dict[str, Path],
    image_paths: Dict[str, Path],
    output_dir: Path,
    exp: _ExportCfg,
) -> None:
    """Phase 3: 读取 LaMa 预测结果，写出最终修复图与可选的 CYWS 格式。"""
    for pred_file in pred_files:
        stem = pred_file.stem.rsplit("_mask", 1)[0]
        with _suppress_stderr(_is_png_warning_quiet()):
            pred_img = cv2.imread(str(pred_file))
        if pred_img is None:
            continue
        mp = current_mask_paths.get(stem)
        if mp is None or not mp.exists():
            continue
        with _suppress_stderr(_is_png_warning_quiet()):
            mask_img = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            continue
        _finalize_one_result(
            stem=stem,
            image_path=image_paths[stem],
            pred_bgr=pred_img,
            mask_u8=mask_img,
            output_dir=output_dir,
            exp=exp,
        )


def run_inpaint_shard(cfg: _RunConfig) -> None:
    """单 shard 入口：准备掩码 → LaMa 批处理 → 收集结果。"""
    random.seed(cfg.seed)
    _safe_mkdir(cfg.output_dir)

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(cfg.ann_file))

    image_files = _slice_by_shard(
        _list_image_files(cfg.image_dir),
        shard_id=cfg.shard_id,
        num_shards=cfg.num_shards,
    )
    if cfg.stem_allowlist is not None:
        image_files = [p for p in image_files if p.stem in cfg.stem_allowlist]
    if not image_files:
        print(f"[shard {cfg.shard_id}] 无图像可处理（分片或 stem 白名单过滤后为空），exit.")
        return

    predict_script = cfg.lama_repo_dir / "bin" / "predict.py"
    if not predict_script.exists():
        raise FileNotFoundError(f"LaMa predict script not found: {predict_script}")
    if not cfg.lama_model_path.exists():
        raise FileNotFoundError(f"LaMa model path not found: {cfg.lama_model_path}")

    chunks = [image_files[i : i + cfg.chunk_size] for i in range(0, len(image_files), cfg.chunk_size)]
    env = os.environ.copy()
    repo_path_str = str(cfg.lama_repo_dir)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = repo_path_str if not existing_pythonpath else f"{repo_path_str}:{existing_pythonpath}"

    exp = _ExportCfg(shard_id=cfg.shard_id, cyws_root=cfg.cyws_root)

    for chunk_idx, chunk_files in enumerate(chunks):
        shard_tag = f"shard{cfg.shard_id}_chunk{chunk_idx}"
        work_dir = cfg.output_dir.parent / f".lama_inputs_{shard_tag}"
        pred_out_dir = cfg.output_dir.parent / f".lama_outputs_{shard_tag}"

        _ensure_empty_dir(work_dir)
        _ensure_empty_dir(pred_out_dir)

        valid_items, mask_paths = _prepare_masks_for_chunk(
            chunk_image_files=chunk_files,
            work_dir=work_dir,
            coco=coco,
            cfg=cfg,
            chunk_idx=chunk_idx,
            num_chunks=len(chunks),
        )
        if not valid_items:
            print(f"[WARN] Shard {cfg.shard_id} chunk {chunk_idx}: no valid inputs, skip.")
            for p in (work_dir, pred_out_dir):
                if p.exists():
                    shutil.rmtree(p)
            continue

        image_paths = {s: ip for s, ip in valid_items}

        code, tail, pred_files = _execute_lama_chunk(
            cfg=cfg,
            work_dir=work_dir,
            pred_out_dir=pred_out_dir,
            env=env,
            chunk_idx=chunk_idx,
            num_chunks=len(chunks),
        )
        if code != 0 and not pred_files:
            logf = pred_out_dir / "predict_stdout.log"
            hint = (
                f"\n完整日志: {logf}\n"
                "常见修复: pip install easydict\n"
                "若 NumPy / PyTorch 报 _ARRAY_API: pip install 'numpy<2'\n"
            )
            raise RuntimeError(
                f"LaMa failed shard={cfg.shard_id} chunk={chunk_idx} code={code}\n{tail}{hint}"
            )

        _collect_chunk_results(
            pred_files=pred_files,
            current_mask_paths=mask_paths,
            image_paths=image_paths,
            output_dir=cfg.output_dir,
            exp=exp,
        )

        if work_dir.exists():
            shutil.rmtree(work_dir)
        if pred_out_dir.exists():
            shutil.rmtree(pred_out_dir)


def _build_worker_args(
    args: argparse.Namespace,
    shard_id: int,
    num_shards: int,
) -> List[str]:
    """构建 worker 子进程的 CLI 参数列表。"""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--data-dir", str(args.data_dir),
        "--split", args.split,
        "--output-dir", str(args.output_dir),
    ]
    if args.cyws_root is not None:
        cmd.extend(["--cyws-root", str(args.cyws_root)])
        cmd.extend(["--cyws-variants-per-image", str(args.cyws_variants_per_image)])
        cmd.extend(["--cyws-unique-iou-thresh", str(args.cyws_unique_iou_thresh)])
        cmd.extend(["--cyws-unique-max-tries", str(args.cyws_unique_max_tries)])
        cmd.extend(["--cyws-train-rate", str(args.cyws_train_rate)])
    cmd.extend([
        "--lama-repo-dir", str(args.lama_repo_dir),
        "--lama-model-path", str(args.lama_model_path),
        "--lama-checkpoint", args.lama_checkpoint,
        "--lama-python", str(args.lama_python),
        "--seed", str(args.seed),
        "--max-mask-area-ratio", str(args.max_mask_area_ratio),
        "--mask-resample-attempts", str(args.mask_resample_attempts),
        "--chunk-size", str(args.chunk_size),
        "--min-coco-ann-area", str(args.min_coco_ann_area),
        "--num-shards", str(num_shards),
        "--shard-id", str(shard_id),
        "--worker",
    ])
    if args.quiet_png_warnings:
        cmd.append("--quiet-png-warnings")
    if getattr(args, "quiet_unique_warnings", False):
        cmd.append("--quiet-unique-warnings")
    if args.stem_allowlist_file is not None:
        cmd.extend(["--stem-allowlist-file", str(args.stem_allowlist_file)])
    return cmd


def _spawn_workers_and_wait(args: argparse.Namespace, gpu_ids: List[str]) -> None:
    num = len(gpu_ids)
    output_dir: Path = args.output_dir
    _safe_mkdir(output_dir)
    image_dir: Path = args.data_dir / args.split
    total_images = len(_list_image_files(image_dir))
    if getattr(args, "cyws_variants_per_image", 1) is not None:
        try:
            v = int(getattr(args, "cyws_variants_per_image", 1))
            if v > 1:
                total_images *= v
        except Exception:
            pass

    procs = []
    for shard_id, gid in enumerate(gpu_ids):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gid
        env["PIPELINE_DISABLE_TQDM"] = "1"
        if args.quiet_png_warnings:
            env["PIPELINE_QUIET_PNG_WARNINGS"] = "1"
        print(f"[multi-gpu] shard {shard_id}/{num} -> CUDA_VISIBLE_DEVICES={gid}")
        procs.append(subprocess.Popen(_build_worker_args(args, shard_id, num), env=env))

    _warn_inpaint_total_pbar_if_residual(output_dir, total_images)
    pbar = _tqdm(total=total_images, desc="Auto-parallel inpaint-export total")
    while any(p.poll() is None for p in procs):
        pbar.n = min(_count_inpainted_jpgs(output_dir), total_images)
        pbar.refresh()
        time.sleep(2.0)

    codes = [p.wait() for p in procs]
    failed = [i for i, c in enumerate(codes) if c != 0]
    pbar.n = min(_count_inpainted_jpgs(output_dir), total_images)
    pbar.refresh()
    pbar.close()
    if failed:
        raise RuntimeError(f"子进程失败 shard indices={failed}, exit codes={codes}")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="COCO 在线掩码 + LaMa 多卡填充；导出为 CYWS(coco-inpainted) 格式数据集。")
    p.add_argument("--data-dir", type=Path, default=Path("./images"))
    p.add_argument("--split", type=str, default="train2017")
    p.add_argument("--output-dir", type=Path, default=None, help="修复图 .jpg，默认 <data-dir>/inpainted_images")
    p.add_argument(
        "--cyws-root",
        type=Path,
        default=None,
        help=(
            "若提供：额外导出为 CYWS(coco-inpainted) 目录结构（images_and_masks/、inpainted/、data_split.pkl），"
            "可直接作为 train/train_change_pairs.py 的 --coco-inpainted-root"
        ),
    )
    p.add_argument(
        "--cyws-variants-per-image",
        type=int,
        default=1,
        help="每张图生成多少个 {idx}_mask{k}（不同掩码/不同填充形式）。=1 时保持原行为（stem 不加 _mask{k}）。",
    )
    p.add_argument(
        "--cyws-unique-iou-thresh",
        type=float,
        default=0.95,
        help="同一张图内多掩码去重的 IoU 阈值；若新 mask 与任一已选 mask 的 IoU ≥ 阈值，则重采样。",
    )
    p.add_argument(
        "--cyws-unique-max-tries",
        type=int,
        default=50,
        help="为生成每个 unique mask 的最大重采样次数（超出则跳过该 mask_id）。",
    )
    p.add_argument(
        "--cyws-train-rate",
        type=float,
        default=0.95,
        help="写 data_split.pkl 时 train 比例（val=1-train_rate）；仅在 --cyws-root 下生效。",
    )
    p.add_argument("--gpu-ids", type=str, default="0,1,2,3")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-mask-area-ratio",
        type=float,
        default=0.7,
        help="掩码前景占全图面积比例上限（默认 0.7，与 CocoMetaDataset 实例交换 exchange_max_area_ratio 对齐）",
    )
    p.add_argument("--mask-resample-attempts", type=int, default=20)
    p.add_argument(
        "--min-coco-ann-area",
        type=int,
        default=0,
        help="忽略 COCO 标注中 area 小于该值的实例（像素²）；0 不过滤，可试 512~4096 减少过碎掩码",
    )
    p.add_argument("--chunk-size", type=int, default=1000)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--quiet-png-warnings", action="store_true")
    p.add_argument(
        "--quiet-unique-warnings",
        action="store_true",
        help="静默输出：不打印 '无法在 N 次内采到 unique mask（跳过该变体）' 这类提示",
    )
    p.add_argument("--lama-repo-dir", type=Path, default=Path("/home/ldl/datasets/lama"))
    p.add_argument("--lama-model-path", type=Path, default=Path("/home/ldl/datasets/lama/big-lama"))
    p.add_argument("--lama-checkpoint", type=str, default="best.ckpt")
    p.add_argument(
        "--lama-python",
        type=str,
        default=None,
        help="运行 LaMa bin/predict.py 的 Python 可执行文件，默认与当前脚本相同（sys.executable），勿再用系统自带的 python3 以免缺依赖",
    )
    p.add_argument(
        "--clean-intermediates",
        action="store_true",
        help="仅删除 LaMa 临时目录 .lama_*，然后退出（扫描路径见 --clean-search-dir 说明）",
    )
    p.add_argument(
        "--clean-search-dir",
        action="append",
        default=[],
        metavar="DIR",
        help="与 --clean-intermediates 联用：额外扫描目录（可重复）。默认始终包含 --data-dir、--output-dir 的父目录、以及本仓库下的 data/（若存在）",
    )
    p.add_argument(
        "--stem-allowlist-file",
        type=Path,
        default=None,
        help="仅处理该文件列出的 stem（一行一个）；多卡时各 shard 与全集求交",
    )
    p.add_argument(
        "--reset-inpaint-data",
        action="store_true",
        help="在跑 LaMa 前删除 output_dir 下已生成的修复图及 change_pairs 的 A/B/label（不删 JSON/annotations；多卡时仅主进程删一次）",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.data_dir / "inpainted_images"
    if args.lama_python is None:
        args.lama_python = sys.executable

    if args.clean_intermediates:
        roots: List[Path] = [args.data_dir, args.output_dir.parent]
        repo_data = _REPO_ROOT / "data"
        if repo_data.is_dir():
            roots.append(repo_data)
        roots.extend(Path(p) for p in args.clean_search_dir)
        n = clean_intermediate_directories(*roots)
        print(f"[clean] 共删除 {n} 个中间目录。", flush=True)
        return

    image_dir = args.data_dir / args.split
    if not image_dir.is_dir():
        raise FileNotFoundError(f"未找到图像目录: {image_dir}")

    ann_file = args.data_dir / "annotations" / f"instances_{args.split}.json"
    if not ann_file.exists():
        raise FileNotFoundError(f"未找到标注: {ann_file}")

    gpu_ids = [g.strip() for g in args.gpu_ids.split(",") if g.strip() != ""]
    if not gpu_ids:
        raise ValueError("--gpu-ids 不能为空")

    if (
        getattr(args, "cyws_variants_per_image", 1) > 1
        and args.mask_resample_attempts > 1
    ):
        print(
            f"[WARN] --mask-resample-attempts={args.mask_resample_attempts} 在 "
            f"--cyws-variants-per-image={args.cyws_variants_per_image} > 1 时无效。"
            "unique 模式下每个候选 mask 仅尝试 1 次重采样（外层由 --cyws-unique-max-tries 控制重试）；"
            "请用 --cyws-unique-max-tries 控制采样开销。",
            flush=True,
        )

    if args.quiet_png_warnings:
        os.environ["PIPELINE_QUIET_PNG_WARNINGS"] = "1"

    stem_allowlist: Optional[Set[str]] = None
    if args.stem_allowlist_file is not None:
        if not args.stem_allowlist_file.is_file():
            raise FileNotFoundError(f"--stem-allowlist-file 不存在: {args.stem_allowlist_file}")
        stem_allowlist = _load_stem_allowlist_file(args.stem_allowlist_file)
        print(f"[allowlist] 已加载 {len(stem_allowlist)} 个 stem", flush=True)

    if args.reset_inpaint_data and not args.worker:
        ni = _delete_inpaint_outputs(args.output_dir)
        print(
            f"[reset-inpaint] 已删除 output_dir 中 {ni} 个文件（output_dir={args.output_dir}）",
            flush=True,
        )

    # 若启用 CYWS 导出：提前写 data_split.pkl（按 idx 列表划分），方便训练侧直接扫描。
    if args.cyws_root is not None and (not args.worker) and (not args.clean_intermediates):
        image_dir = args.data_dir / args.split
        stems = [p.stem for p in _list_image_files(image_dir)]
        if stem_allowlist is not None:
            stems = [s for s in stems if s in stem_allowlist]
        idxs: List[int] = []
        for s in stems:
            if s.isdigit():
                idxs.append(int(s))
        idxs = sorted(set(idxs))
        rng = random.Random(int(args.seed))
        rng.shuffle(idxs)
        train_n = int(round(float(args.cyws_train_rate) * len(idxs)))
        train = sorted(idxs[:train_n])
        val = sorted(idxs[train_n:])
        cyws_root = Path(args.cyws_root).expanduser().resolve()
        _safe_mkdir(cyws_root)
        with (cyws_root / "data_split.pkl").open("wb") as f:
            pickle.dump({"train": train, "val": val}, f)
        print(
            f"[cyws] 已写 data_split.pkl：train={len(train)} val={len(val)} root={cyws_root}",
            flush=True,
        )

    if not args.worker and len(gpu_ids) > 1:
        _spawn_workers_and_wait(args, gpu_ids)
        return

    if not args.worker and len(gpu_ids) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]
        total = len(_list_image_files(args.data_dir / args.split))
        try:
            v = int(getattr(args, "cyws_variants_per_image", 1))
            if v > 1:
                total *= v
        except Exception:
            pass
        _warn_inpaint_total_pbar_if_residual(
            args.output_dir, total
        )

    cfg = _RunConfig(
        ann_file=ann_file,
        image_dir=image_dir,
        output_dir=args.output_dir,
        lama_repo_dir=args.lama_repo_dir,
        lama_model_path=args.lama_model_path,
        lama_checkpoint=args.lama_checkpoint,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        seed=args.seed,
        max_mask_area_ratio=args.max_mask_area_ratio,
        mask_resample_attempts=args.mask_resample_attempts,
        chunk_size=args.chunk_size,
        lama_python=args.lama_python,
        quiet_unique_warnings=bool(getattr(args, "quiet_unique_warnings", False)),
        min_coco_ann_area=args.min_coco_ann_area,
        stem_allowlist=stem_allowlist,
        cyws_root=args.cyws_root,
        cyws_variants_per_image=int(args.cyws_variants_per_image),
        cyws_unique_iou_thresh=float(args.cyws_unique_iou_thresh),
        cyws_unique_max_tries=int(args.cyws_unique_max_tries),
    )
    run_inpaint_shard(cfg)


if __name__ == "__main__":
    main()
