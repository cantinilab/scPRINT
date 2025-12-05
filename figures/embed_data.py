import base64
import os
import re

html_path = "/pasteur/appa/homes/jkalfon/scPRINT/figures/nice_umap_scprint2.html"
base_dir = "/pasteur/appa/homes/jkalfon/scPRINT/figures/"

files = {
    "pointDataEncoded": "datamapplot_point_data_0.zip",
    "hoverDataEncoded": "datamapplot_meta_data_0.zip",
    "labelDataEncoded": "datamapplot_label_data.zip",
}

with open(html_path, "r") as f:
    content = f.read()

for var_name, filename in files.items():
    file_path = os.path.join(base_dir, filename)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    with open(file_path, "rb") as f:
        data = f.read()
        b64_data = base64.b64encode(data).decode("utf-8")
        data_uri = f"data:application/zip;base64,{b64_data}"

    # Regex to match the variable declaration
    # const pointDataEncoded = [
    #   `${directoryPath}datamapplot_point_data_0.zip`,
    #   ];

    # We use a non-greedy match for the content inside brackets
    pattern = re.compile(f"const {var_name} = \[.*?\];", re.DOTALL)

    replacement = f"const {var_name} = [`{data_uri}`];"

    if pattern.search(content):
        print(f"Replacing {var_name}")
        content = pattern.sub(replacement, content)
    else:
        print(f"Pattern not found for {var_name}")

with open(html_path, "w") as f:
    f.write(content)
