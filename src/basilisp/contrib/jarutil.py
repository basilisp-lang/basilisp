import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile


def install_jar(site_packages: Path, source_jar: Path) -> None:
    if source_jar.suffix != ".jar":
        raise ValueError(f"Expected a *.jar file; got *{source_jar.suffix}")

    with TemporaryDirectory() as tmpdir, ZipFile(source_jar, mode="r") as zf:
        zf.extractall(
            path=tmpdir,
            members=[mem for mem in zf.namelist() if not mem.startswith("META-INF")],
        )
        for p in Path(tmpdir).iterdir():
            if p.is_dir():
                destination = site_packages / p.name
                d = shutil.copytree(p, destination)
                print(f"Created {d}")
            else:
                f = shutil.copy(p, site_packages)
                print(f"Created {f}")
