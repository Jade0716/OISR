import os
import sys
from argparse import ArgumentParser
from tqdm import tqdm

sys.path.append('./utils')
from utils.config_utils import PARTNET_ID_PATH, PARTNET_CAMERA_POSITION_RANGE

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--ray_tracing', type=bool, default=False,
                        help='Specify whether to use ray tracing in rendering')
    parser.add_argument('--replace_texture', type=bool, default=False,
                        help='Specify whether to replace the texture of the rendered image using the original model')
    parser.add_argument('--start_idx', type=int, default=0, help='Specify the start index of the model id to render')
    parser.add_argument('--num_render', type=int, default=32,
                        help='Specify the number of renderings for each model id each camera range')
    parser.add_argument('--log_dir', type=str, default='/16T/liuyuyan/example_rendered/log_render.txt', help='Specify the log file')

    args = parser.parse_args()

    ray_tracing = args.ray_tracing
    replace_texture = args.replace_texture
    start_idx = args.start_idx
    num_render = args.num_render
    log_dir = args.log_dir

    model_id_list = []
    with open(PARTNET_ID_PATH, 'r') as fd:
        for line in fd:
            ls = line.strip().split(' ')
            model_id_list.append((ls[0], int(ls[1])))

    total_to_render = len(model_id_list)
    cnt = 0

    for category, model_id in tqdm(model_id_list, desc="Rendering models", unit="model", leave=True):
        # print(f'Still to render: {total_to_render - cnt}\n')
        if category in ["Box", "Bucket", "Camera", "CoffeeMachine", "Dishwasher", "Door"]:
        # if category in ["Keyboard", "KitchenPot", "Laptop", "Microwave", "Oven", "Phone"]:
        # if category in ["Printer", "Remote", "Safe", "StorageFurniture"]:
        # if category in ["Suitcase", "Table", "Toaster", "Toilet", "TrashCan", "WashingMachine", "Refrigerator"]:

        # if category in ["Box"]:
            for pos_idx in range(len(PARTNET_CAMERA_POSITION_RANGE[category])):
                for render_idx in  range(num_render):
                    # print(f'Rendering: {category} : {model_id} : {pos_idx} : {start_idx + render_idx}\n')

                    render_string = f'CUDA_VISIBLE_DEVICES=0 python -u render.py --dataset partnet --model_id {model_id} --camera_idx {pos_idx} --render_idx {start_idx + render_idx}'
                    if ray_tracing:
                        render_string += ' --ray_tracing True'
                    if replace_texture:
                        render_string += ' --replace_texture True'
                    # render_string += f' 2>&1 | tee -a {log_dir}'

                    os.system(render_string)

            print(f'Render Over: {category} : {model_id}\n')
            cnt += 1

    print("Over!!!")