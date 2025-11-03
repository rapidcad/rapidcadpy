from pathlib import Path
import runpy
import traceback
import sys

ROOT = r"C:\\Users\\Administrator\\rapidcad_data"
DEST_ROOT = r"C:\\Users\\Administrator\\deepcad_ipt"


def main():
    root = Path(ROOT)
    dest_root = Path(DEST_ROOT)
    if not root.exists():
        print(f"Root not found: {root}")
        return 1

    files = [p for p in root.rglob("*.py") if p.name != "__init__.py"]
    if not files:
        print(f"No python files found under: {root}")
        return 1

    total = len(files)
    ok_count = 0
    fail_count = 0

    print(f"Discovered {total} files. Starting execution…")

    for idx, fpath in enumerate(sorted(files), start=1):
        rel_path = fpath.relative_to(root)
        print(f"[{idx}/{total}] Running: {rel_path.as_posix()}")
        try:
            # Execute and capture globals
            g = runpy.run_path(str(fpath), run_name="__main__")
            # Try to export using 'app' from the script
            app = g.get("app")
            if app is not None:
                ipt_out = (dest_root / rel_path).with_suffix(".ipt")
                # Ensure destination folder exists
                ipt_out.parent.mkdir(parents=True, exist_ok=True)
                to_ipt = getattr(app, "to_ipt", None)
                close_document = getattr(app, "close_document", None)
                if callable(to_ipt):
                    try:
                        to_ipt(str(ipt_out))
                        print(f"  -> Saved IPT: {ipt_out}")
                    except Exception as e:
                        print(f"  -> to_ipt failed: {e}")
                if callable(close_document):
                    try:
                        close_document()
                    except Exception as e:
                        print(f"  -> close_document failed: {e}")
            ok_count += 1
        except SystemExit as se:
            code = getattr(se, "code", 0) or 0
            if code == 0:
                ok_count += 1
            else:
                fail_count += 1
                print(f"  -> SystemExit with code {code}")
        except Exception:
            fail_count += 1
            print("  -> ERROR:\n" + traceback.format_exc())

    print(f"Done. OK: {ok_count}, Fail: {fail_count}")
    return 0 if fail_count == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
