import os
import xml.etree.ElementTree as ET

category_rules = {
    # "Box":("hinge lid", "free box_body"),
    # "Refrigerator":("hinge door", "heavy refrigerator_body"),
    # "Suitcase": ("hinge lid", "free suitcase_body"),
    "Toilet": ("slider lid", "static toilet_body"),


    "TrashCan": ("slider lid", "free trashcan_body"),
    # "KitchenPot": ("slider lid", "free pot_body"),
    "CoffeeMachine": ("slider lid", "free coffee_machine_body"),
}

base_dir = "/16T/liuyuyan/partnet_mobility_part"
id_list_file = "/home/liuyuyan/GaPartNet/dataset/render_tools/meta/partnet_all_id_list.txt"

category_to_ids = {}
with open(id_list_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2 and parts[0] in category_rules:
            category_to_ids.setdefault(parts[0], []).append(parts[1])

def modify_urdf(model_dir, slider_key, static_key):
    semantics_file = os.path.join(model_dir, "semantics_gapartnet.txt")
    urdf_file = os.path.join(model_dir, "mobility_annotation_gapartnet.urdf")
    if not os.path.exists(semantics_file) or not os.path.exists(urdf_file):
        return False

    name2link = {}
    with open(semantics_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                link = parts[0]
                joint_type = parts[1]
                sem_name = parts[2]
                full_name = f"{joint_type} {sem_name}"
                name2link[full_name] = link

    slider_link = name2link.get(slider_key)
    static_link = name2link.get(static_key)
    if not slider_link or not static_link:
        return False

    tree = ET.parse(urdf_file)
    root = tree.getroot()
    modified = False

    for joint in root.findall("joint"):
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        if (parent == slider_link and child == static_link) or (child == slider_link and parent == static_link):
            limit = joint.find("limit")
            if limit is not None and "upper" in limit.attrib:
                try:
                    original = float(limit.attrib["upper"])
                    limit.attrib["upper"] = str(original * 10)
                    modified = True
                except:
                    pass


    if modified:
        tree.write(urdf_file)
    return modified

total = 0
modified_count = 0

for category, ids in category_to_ids.items():
    slider_key, static_key = category_rules[category]
    for model_id in ids:
        model_dir = os.path.join(base_dir, model_id)
        total += 1
        if modify_urdf(model_dir, slider_key, static_key):
            modified_count += 1

print(f"Total matched objects: {total}")
print(f"Successfully modified: {modified_count}")
