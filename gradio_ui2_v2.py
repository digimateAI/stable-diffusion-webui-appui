import pandas as pd
import requests
import base64


def check_for_new_rows(file_path,parent_folder,new_folder_name):
    try:
        # # Load the Excel file into a pandas DataFrame
        # df = pd.read_excel(file_path)
        # print(len(df))

        # fourth_last_index = df.index[-4]
        # third_last_index = df.index[-3]
        # second_last_index = df.index[-2]
        # last_index = df.index[-1]
        # # Get the last four rows (newly added rows)
        # fourth_last_row = df.iloc[-4]
        # third_last_row = df.iloc[-3]
        # second_last_row = df.iloc[-2]
        # last_row = df.iloc[-1]

        # Define the URL and the payload to send.
        url = "http://127.0.0.1:7860"

        # Define important values from the last new row
        payload1 = {'prompt': 'Ultra-realistic extremely detailed real life 8k masterpiece of VIRAT a man, extremely detailed facial features  <lora:VIRAT-LORA_img10_rep13_noReg_nd32e5:1>, (soothing tones:1.25), (hdr:1.25), (artstation:1.2), dramatic, (intricate details:1.14), (hyperrealistic 3d render:1.16), (filmic:0.55), (rutkowski:1.1), (faded:1.3)', 'negative_prompt': '(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, bad eyes, crossed eyes, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation', 'seed': 1186160619, 'width': 1024, 'height': 1024, 'sampler_name': 'Euler a', 'cfg_scale': 9, 'steps': 35}
        payload2 = {'prompt': 'Ultra-realistic extremely detailed real life 8k masterpiece of VIRAT a man, extremely detailed facial features  <lora:VIRAT-LORA_img10_rep13_noReg_nd32e5:1>, (soothing tones:1.25), (hdr:1.25), (artstation:1.2), dramatic, (intricate details:1.14), (hyperrealistic 3d render:1.16), (filmic:0.55), (rutkowski:1.1), (faded:1.3)', 'negative_prompt': '(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, bad eyes, crossed eyes, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation', 'seed': 1186160619, 'width': 1024, 'height': 1024, 'sampler_name': 'Euler a', 'cfg_scale': 9, 'steps': 35}
        payload3 = {'prompt': 'Ultra-realistic extremely detailed real life 8k masterpiece of VIRAT a man, extremely detailed facial features  <lora:VIRAT-LORA_img10_rep13_noReg_nd32e5:1>, (soothing tones:1.25), (hdr:1.25), (artstation:1.2), dramatic, (intricate details:1.14), (hyperrealistic 3d render:1.16), (filmic:0.55), (rutkowski:1.1), (faded:1.3)', 'negative_prompt': '(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, bad eyes, crossed eyes, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation', 'seed': 1186160619, 'width': 1024, 'height': 1024, 'sampler_name': 'Euler a', 'cfg_scale': 9, 'steps': 35}
        payload4 = {'prompt': 'Ultra-realistic extremely detailed real life 8k masterpiece of VIRAT a man, extremely detailed facial features  <lora:VIRAT-LORA_img10_rep13_noReg_nd32e5:1>, (soothing tones:1.25), (hdr:1.25), (artstation:1.2), dramatic, (intricate details:1.14), (hyperrealistic 3d render:1.16), (filmic:0.55), (rutkowski:1.1), (faded:1.3)', 'negative_prompt': '(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, bad eyes, crossed eyes, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation', 'seed': 1186160619, 'width': 1024, 'height': 1024, 'sampler_name': 'Euler a', 'cfg_scale': 9, 'steps': 35}
        # print(payload1)

        # Send said payload to said URL through the API.
        response1 = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload1)
        response2 = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload2)
        response3 = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload3)
        response4 = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload4)
        r1 = response1.json()
        r2 = response2.json()
        r3 = response3.json()
        r4 = response4.json()

        imgfile_paths = []
        # Decode the responses and save the images.
        # Loop through each response and save the corresponding image
        for i, response in enumerate([r1, r2, r3, r4], start=1):
            index_value = {1: 'fourth_last_index', 2: 'third_last_index', 3: 'second_last_index'}.get(i, 'last_index')
            image_data = response['images'][0]
            filename = f"output/txt2img/VIRAT_{(index_value+1)}_{i}.png"
            
            imgfile_paths.append(filename)

            with open(filename, 'wb') as f:
                f.write(base64.b64decode(image_data))

        return imgfile_paths



from PIL import Image
def display_images(username, gender, select_option):
    parent_folder = "output/txt2img/"
    # file_path = parent_folder+username+"/prompt_excel_files/settings.xlsx"
    file_path = "img_gen_settings.xlsx"
    # append_to_excel(file_path, select_option, username, gender)
    imgpaths = check_for_new_rows(file_path,parent_folder,username)
    
    # images = [cv2.imread(path) for path in imgpaths]
    images = [Image.open(path) for path in imgpaths]
    # # Combine images into a single numpy array
    # combined_image = np.concatenate(images, axis=1)
    # return combined_image
    # # Convert images to numpy arrays
    # images_array = np.array(images)
    # return images_array
    # # Convert images to Gradio-friendly format
    # gr_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
    # return gr_images
    return images









import gradio as gr
dropdown_options = ["portrait-1", "portrait-2"]
dropdown_gender = ["man", "woman"]

iface = gr.Interface(fn=display_images,
                    #  inputs=["text","text","text"],
                     inputs=[gr.Textbox(label="Username"), gr.Dropdown(label="Gender", choices=dropdown_gender), gr.Dropdown(label="Select Option", choices=dropdown_options)],
                     outputs=["image", "image", "image", "image"],
                     title="Create Your Own Images")
iface.launch()



