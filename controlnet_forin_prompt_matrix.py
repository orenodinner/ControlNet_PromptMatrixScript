import modules.scripts as scripts
import gradio as gr
import os
import re

import modules.shared as shared
import modules.scripts as scripts
import modules.sd_samplers
from modules.processing import process_images, StableDiffusionProcessingTxt2Img

import copy
import os
import shutil

import cv2
import gradio as gr
import modules.scripts as scripts

from modules import images
from modules.processing import process_images
from modules.shared import opts
from PIL import Image


# リスト形式の全ての画像をcv2で読み込む
def read_images(path):
    img_list = []
    for img_path in path:
        img = cv2.imread(img_path.name)
        img_list.append(img)
    return img_list


# リストのCV2画像をフォルダーに保存する。保存名は000000.pngから順に保存される。
def save_images(img_list, path):
    for i, img in enumerate(img_list):
        cv2.imwrite(os.path.join(path, f"{i:06}.png"), img)


def prompt_matrix(p):
    modules.processing.fix_seed(p)

    original_prompt = p.prompt[0] if type(p.prompt) == list else p.prompt

    matrix_count = 0
    prompt_matrix_parts = []
    for data in re.finditer(r'(<([^>]+)>)', original_prompt):
        if data:
            matrix_count += 1
            span = data.span(1)
            items = data.group(2).split("|")
            prompt_matrix_parts.extend(items)

    all_prompts = [original_prompt]
    while True:
        found_matrix = False
        for this_prompt in all_prompts:
            for data in re.finditer(r'(<([^>]+)>)', this_prompt):
                if data:
                    found_matrix = True
                    # Remove last prompt as it has a found_matrix
                    all_prompts.remove(this_prompt)
                    span = data.span(1)
                    items = data.group(2).split("|")
                    for item in items:
                        new_prompt = this_prompt[:span[0]] + \
                            item.strip() + this_prompt[span[1]:]
                        all_prompts.append(new_prompt.strip())
                break
            if found_matrix:
                break
        if not found_matrix:
            break

    total_images = len(all_prompts) * p.n_iter
    print(f"Prompt matrix will create {total_images} images")

    total_steps = p.steps * total_images
    if isinstance(p, StableDiffusionProcessingTxt2Img) and p.enable_hr:
        total_steps *= 2
    shared.total_tqdm.updateTotal(total_steps)

    p.prompt = all_prompts * p.n_iter
    p.seed = [item for item in range(int(p.seed), int(
        p.seed) + p.n_iter) for _ in range(len(all_prompts))]
    p.n_iter = total_images
    p.prompt_for_display = original_prompt

    return process_images(p)


class Script(scripts.Script):
    def title(self):
        return "Controlnet forin prompt matrix"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        # How the script's is displayed in the UI. See https://gradio.app/docs/#components
        # for the different UI components you can use and how to create them.
        # Most UI components can return a value, such as a boolean for a checkbox.
        # The returned values are passed to the run method as parameters.

        # Create an empty tuple to store the controls.
        ctrls_group = ()
        # Get the maximum number of models to display from the options.
        max_models = opts.data.get("control_net_max_models_num", 1)

        # Create a group to hold the controls.
        with gr.Group():
            # Create an accordion to hold the tabs.
            with gr.Accordion("ControlNet-Forin-Prompt", open=False):
                # Create a tab set.
                with gr.Tabs():
                    # Create a tab for each model.
                    for i in range(max_models):
                        with gr.Tab(f"ControlNet-{i}", open=False):
                            # Add a video player for each model.
                            ctrls_group += (gr.Files(file_count="directory",
                                            source='upload', elem_id=f"directroy_{i}"), )

        # Return the controls group.
        # print("ctrls_group", ctrls_group)
        return ctrls_group

    def run(self, p, image_list):

        output_image_list = []
        imagess = read_images(image_list)
        for im in imagess:
            print("im", im)
            copy_p = copy.copy(p)
            copy_p.control_net_input_image = []
            copy_p.control_net_input_image.append(im)
            proc = prompt_matrix(copy_p)
            img = proc.images[0]
            output_image_list.append(img)
            copy_p.close()

        return proc
