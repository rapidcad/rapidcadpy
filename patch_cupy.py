import os

file_path = "/opt/conda/envs/cadgpt/lib/python3.10/site-packages/cupy/_environment.py"

with open(file_path, "r") as f:
    content = f.read()

old_code = """    installed_names = {d.metadata.get("Name", None)
                       for d in importlib.metadata.distributions()}"""

new_code = """    installed_names = {d.metadata.get("Name", None)
                       for d in importlib.metadata.distributions() if d.metadata is not None}"""

if old_code in content:
    new_content = content.replace(old_code, new_code)
    with open(file_path, "w") as f:
        f.write(new_content)
    print("Successfully patched cupy/_environment.py")
else:
    print(
        "Could not find the code to patch. It might have been already patched or different."
    )
    # Print the section to debug
    start_idx = content.find("def _detect_duplicate_installation():")
    if start_idx != -1:
        print("Found function, printing first 500 chars:")
        print(content[start_idx : start_idx + 500])
