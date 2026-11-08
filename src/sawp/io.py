from __future__ import annotations

import ast
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ArrayMember:
    name: str
    shape: tuple[int, ...]
    dtype: str
    fortran_order: bool
    compressed_bytes: int
    uncompressed_bytes: int


def _npy_header(stream) -> tuple[tuple[int, ...], str, bool]:
    if stream.read(6) != b"\x93NUMPY":
        raise ValueError("invalid NPY member")
    major, _minor = stream.read(2)
    length = struct.unpack("<H" if major == 1 else "<I", stream.read(2 if major == 1 else 4))[0]
    header = ast.literal_eval(stream.read(length).decode("latin1").strip())
    return tuple(header["shape"]), str(header["descr"]), bool(header["fortran_order"])


def inspect_npz(path: str | Path) -> dict[str, ArrayMember]:
    archive_path = Path(path)
    members: dict[str, ArrayMember] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if not info.filename.endswith(".npy"):
                continue
            with archive.open(info) as stream:
                shape, dtype, fortran = _npy_header(stream)
            name = Path(info.filename).stem
            members[name] = ArrayMember(
                name=name,
                shape=shape,
                dtype=dtype,
                fortran_order=fortran,
                compressed_bytes=info.compress_size,
                uncompressed_bytes=info.file_size,
            )
    return members


def open_npz_memmap(
    path: str | Path,
    member: str = "data",
    *,
    mode: str = "r",
) -> np.memmap:
    """Memory-map an uncompressed NPY member stored inside an NPZ archive.

    ``numpy.load(..., mmap_mode=...)`` cannot memory-map NPZ members. The historical
    SAWP products were written with ``numpy.savez`` and are ZIP_STORED, so their array
    bytes can be mapped in place without loading a 2-4 GB cube into RAM.
    """
    archive_path = Path(path)
    member_name = member if member.endswith(".npy") else f"{member}.npy"
    with zipfile.ZipFile(archive_path) as archive:
        info = archive.getinfo(member_name)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(
                f"{member_name} is compressed; migrate or extract it before memory mapping"
            )
        if info.flag_bits & 0x1:
            raise ValueError("encrypted NPZ members cannot be memory mapped")
        local_header_offset = info.header_offset

    with archive_path.open("rb") as stream:
        stream.seek(local_header_offset + 26)
        filename_length, extra_length = struct.unpack("<HH", stream.read(4))
        npy_offset = local_header_offset + 30 + filename_length + extra_length
        stream.seek(npy_offset)
        if stream.read(6) != b"\x93NUMPY":
            raise ValueError("invalid NPY member")
        major, _minor = stream.read(2)
        length_size = 2 if major == 1 else 4
        header_length = struct.unpack("<H" if major == 1 else "<I", stream.read(length_size))[0]
        header = ast.literal_eval(stream.read(header_length).decode("latin1").strip())
        array_offset = npy_offset + 8 + length_size + header_length
    dtype = np.dtype(header["descr"])
    if dtype.hasobject:
        raise ValueError("object arrays cannot be memory mapped safely")
    return np.memmap(
        archive_path,
        dtype=dtype,
        mode=mode,
        offset=array_offset,
        shape=tuple(header["shape"]),
        order="F" if header["fortran_order"] else "C",
    )


def load_npz(
    path: str | Path,
    *,
    required: tuple[str, ...] = ("data",),
    allow_pickle: bool = False,
    members: tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    archive_path = Path(path)
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    with np.load(archive_path, allow_pickle=allow_pickle) as archive:
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise ValueError(f"{archive_path} is missing members: {', '.join(missing)}")
        selected = archive.files if members is None else members
        unknown = [name for name in selected if name not in archive.files]
        if unknown:
            raise ValueError(f"{archive_path} has no members: {', '.join(unknown)}")
        return {name: archive[name] for name in selected}


def open_dataset(path: str | Path, *, chunks: str | dict = "auto"):
    import xarray as xr

    return xr.open_dataset(Path(path), chunks=chunks)
