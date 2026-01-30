import importlib.metadata


def check_distributions():
    print("Checking distributions...")
    count = 0
    for d in importlib.metadata.distributions():
        count += 1
        # print(f"Checking {d}")
        try:
            meta = d.metadata
            if meta is None:
                print(f"Found distribution with None metadata: {d}")
                print(f"  Type of d: {type(d)}")
                try:
                    print(f"  files: {d.files}")
                except:
                    pass
            else:
                # This is what cupy does
                try:
                    name = meta.get("Name", None)
                except AttributeError:
                    print(
                        f"AttributeError accessing metadata.get for distribution {d}. metadata is {type(meta)}"
                    )
        except Exception as e:
            print(f"Error checking distribution {d}: {e}")
    print(f"Checked {count} distributions.")


if __name__ == "__main__":
    check_distributions()
