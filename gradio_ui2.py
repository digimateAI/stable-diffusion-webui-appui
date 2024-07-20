import pandas as pd
import requests
import base64

def check_for_new_rows(file_path,parent_folder,new_folder_name):
    try:
        # Load the Excel file into a pandas DataFrame
        df = pd.read_excel(file_path)
        print(len(df))

        fourth_last_index = df.index[-4]
        third_last_index = df.index[-3]
        second_last_index = df.index[-2]
        last_index = df.index[-1]
        # Get the last four rows (newly added rows)
        fourth_last_row = df.iloc[-4]
        third_last_row = df.iloc[-3]
        second_last_row = df.iloc[-2]
        last_row = df.iloc[-1]

        # Define the URL and the payload to send.
        url = "http://127.0.0.1:7860"

        # Define important values from the last new row
        payload1 = {"prompt": fourth_last_row['prompt'],"negative_prompt": fourth_last_row['negative_prompt'],"seed": int(fourth_last_row['seed']),"width": int(fourth_last_row['width']),"height": int(fourth_last_row['height']),"sampler_name": fourth_last_row['sampler_name'],"cfg_scale": int(fourth_last_row['cfg_scale']),"steps": int(fourth_last_row['steps'])}
        payload2 = {"prompt": third_last_row['prompt'],"negative_prompt": third_last_row['negative_prompt'],"seed": int(third_last_row['seed']),"width": int(third_last_row['width']),"height": int(third_last_row['height']),"sampler_name": third_last_row['sampler_name'],"cfg_scale": int(third_last_row['cfg_scale']),"steps": int(third_last_row['steps'])}
        payload3 = {"prompt": second_last_row['prompt'],"negative_prompt": second_last_row['negative_prompt'],"seed": int(second_last_row['seed']),"width": int(second_last_row['width']),"height": int(second_last_row['height']),"sampler_name": second_last_row['sampler_name'],"cfg_scale": int(second_last_row['cfg_scale']),"steps": int(second_last_row['steps'])}
        payload4 = {"prompt": last_row['prompt'],"negative_prompt": last_row['negative_prompt'],"seed": int(last_row['seed']),"width": int(last_row['width']),"height": int(last_row['height']),"sampler_name": last_row['sampler_name'],"cfg_scale": int(last_row['cfg_scale']),"steps": int(last_row['steps'])}
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
            index_value = {1: fourth_last_index, 2: third_last_index, 3: second_last_index}.get(i, last_index)
            image_data = response['images'][0]
            filename = f"{parent_folder}{new_folder_name}/output/{new_folder_name}_{(index_value+1)}_{i}.png"
            
            imgfile_paths.append(filename)

            with open(filename, 'wb') as f:
                f.write(base64.b64decode(image_data))

        return imgfile_paths
        
    except Exception as e:
        print("Error:", e)
        return None

################################################################################################################

def generate_prompts(select_option,new_folder_name,class_trigger_word):
    version = f'lora:{new_folder_name}-LORA_img10_rep13_noReg_nd32e5:1'

    # Determine the number of rows
    num_rows = 4  # You mentioned that you want 4 rows for each version

    # Generate values for Changeable_Column1
    changeable_values = [f'{version}'] + [f'{version}.{i}' for i in range(1, num_rows)]

    if select_option == "portrait-1":
        seed = 1186160619   # Replace the seed value for certain photo type.
        prompts = [f'Ultra-realistic extremely detailed real life 8k masterpiece of {new_folder_name} a {class_trigger_word}, extremely detailed facial features  <{idx}>, (soothing tones:1.25), (hdr:1.25), (artstation:1.2), dramatic, (intricate details:1.14), (hyperrealistic 3d render:1.16), (filmic:0.55), (rutkowski:1.1), (faded:1.3)' for idx in changeable_values]
        negative_prompt = '(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, bad eyes, crossed eyes, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation'
    elif select_option == "portrait-2":
        seed = 2389472027
        prompts = [f'Ultra-realistic extremely detailed real life 8k masterpiece of {new_folder_name} a {class_trigger_word}, extremely detailed facial features  <{idx}>, (soothing tones:1.25), (hdr:1.25), (artstation:1.2), dramatic, (intricate details:1.14), (hyperrealistic 3d render:1.16), (filmic:0.55), (rutkowski:1.1), (faded:1.3)' for idx in changeable_values]
        negative_prompt = '(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, bad eyes, crossed eyes, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation'
    else:
        # Handle other cases if needed
        seed = None
        prompts = []
        negative_prompt = None

    return seed, prompts, negative_prompt

def append_to_excel(file_path, select_option, new_folder_name, class_trigger_word):
    # Load the existing DataFrame from the Excel file
    df = pd.read_excel(file_path)

    # Generate four prompts based on the seed
    seed, prompts, negative_prompt = generate_prompts(select_option,new_folder_name,class_trigger_word)

    # Define fixed values for other columns
    fixed_values = {
        'width': 1024,
        'height': 1024,
        'sampler_name': 'Euler a',
        'cfg_scale': 9,
        'steps': 35
    }

    # Create a DataFrame for the new rows
    new_rows = []
    for prompt in prompts:
        # Add prompt value to the fixed values
        fixed_values['prompt'] = prompt
        # Add negative_prompt value to the fixed values
        fixed_values['negative_prompt'] = negative_prompt
        # Add seed value to the fixed values
        fixed_values['seed'] = seed
        # Append fixed values to the new row list
        new_rows.append(fixed_values.copy())

    # Concatenate the new rows with the existing DataFrame
    new_df = pd.DataFrame(new_rows)
    df = pd.concat([df, new_df], ignore_index=True)

    # Write the updated DataFrame back to the Excel file
    df.to_excel(file_path, index=False)

################################################################################################################
# import cv2
from PIL import Image
def display_images(username, gender, select_option):
    parent_folder = "A:/Lora_output/"
    file_path = parent_folder+username+"/prompt_excel_files/settings.xlsx"
    append_to_excel(file_path, select_option, username, gender)
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

################################################################################################################
import gradio as gr
dropdown_options = ["portrait-1", "portrait-2"]
dropdown_gender = ["man", "woman"]

iface = gr.Interface(fn=display_images,
                    #  inputs=["text","text","text"],
                     inputs=[gr.Textbox(label="Username"), gr.Dropdown(label="Gender", choices=dropdown_gender), gr.Dropdown(label="Select Option", choices=dropdown_options)],
                     outputs=["image", "image", "image", "image"],
                     title="Create Your Own Images")
iface.launch()

# import gradio as gr

# img_paths = ["A:/Lora_output/DDIP/output/DDIP_49_1.png","A:/Lora_output/DDIP/output/DDIP_50_2.png","A:/Lora_output/DDIP/output/DDIP_51_3.png","A:/Lora_output/DDIP/output/DDIP_52_4.png"]

# with gr.Blocks() as demo:
#     with gr.Row():
#         with gr.Column(scale=2, min_width=50):
#             img1 = gr.Image(img_paths[0])
#             img2 = gr.Image(img_paths[1])
#             # btn = gr.Button("Go")
#     with gr.Row():
#         with gr.Column(scale=2, min_width=50):
#             img3 = gr.Image(img_paths[2])
#             img4 = gr.Image(img_paths[3])

# demo.launch()