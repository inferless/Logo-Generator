# Logo Generator
- Welcome to this hands-on tutorial, where you'll learn to build a serverless Logo Generator application capable of creating unique logos based on text descriptions.
- We will leverage the Diffusers library, a powerful Python library for text-to-image and image-to-image generation tasks, and Inferless, a cutting-edge platform for deploying and serving machine learning models efficiently.
## Architecture
![Architecture Diagram](https://i.postimg.cc/sg900tVS/Untitled-design-6.png)
---
## Sample Logos
Here are some of the sample logo's generated using this application.
<table>
  <tr>
    <td>
      <img src="https://i.postimg.cc/ZnY1qW9H/image-4.png" alt="Image 1" style="width: 100%;">
      <p style="text-align: center;"><strong>Prompt:</strong> " logo for a cat grooming"</p>
      <p style="text-align: center;"><strong>Negative prompt:</strong> "bad art, ugly, deformed, watermark, duplicated, multiple images"</p>
      <p style="text-align: center;"><strong>Color:</strong> "colorful, bright"</p>
      <p style="text-align: center;"><strong>Inference Time:</strong> 5.30 sec</p>
    </td>
    <td>
      <img src="https://i.postimg.cc/bNMS1Mx0/image-2.png" alt="Image 2" style="width: 100%;">
      <p style="text-align: center;"><strong>Prompt:</strong> “ logo for spacecraft company”</p>
      <p style="text-align: center;"><strong>Negative prompt:</strong> "bad art, ugly, deformed, watermark, duplicated, multiple images"</p>
      <p style="text-align: center;"><strong>Color:</strong> "orange, yellow, black, purple"</p>
      <p style="text-align: center;"><strong>Inference Time:</strong> 5.29 sec</p>
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://i.postimg.cc/k4fzHWwC/image-6.png" alt="Image 4" style="width: 100%;">
      <p style="text-align: center;"><strong>Prompt:</strong> " logo for a biker group"</p>
      <p style="text-align: center;"><strong>Negative prompt:</strong> "bad art, ugly, deformed, watermark, duplicated, multiple images, skull"</p>
      <p style="text-align: center;"><strong>Color:</strong> "bright, powerful, multi colour"</p>
      <p style="text-align: center;"><strong>Inference Time:</strong> 5.39 sec</p>
    </td>
    <td>
      <img src="https://i.postimg.cc/cHxkS65c/image-5.png" alt="Image 3" style="width: 100%;">
      <p style="text-align: center;"><strong>Prompt:</strong> " logo for a dog hospital"</p>
      <p style="text-align: center;"><strong>Negative prompt:</strong> "bad art, ugly, deformed, watermark, duplicated, multiple images"</p>
      <p style="text-align: center;"><strong>Color:</strong> "minimalist"</p>
      <p style="text-align: center;"><strong>Inference Time:</strong> 5.16 sec</p>
    </td>
  </tr>
</table>

---
## Prerequisites
- **Git**. You would need git installed on your system if you wish to customize the repo after forking.
- **Python>=3.8**. You would need Python to customize the code in the app.py according to your needs.
- **Curl**. You would need Curl if you want to make API calls from the terminal itself.

## Quick Start
Here is a quick start to help you get up and running with this template on Inferless.

### Fork the Repository
Get started by forking the repository. You can do this by clicking on the fork button in the top right corner of the repository page.

This will create a copy of the repository in your own GitHub account, allowing you to make changes and customize it according to your needs.

### Create a Custom Runtime in Inferless
To access the custom runtime window in Inferless, simply navigate to the sidebar and click on the **Create new Runtime** button. A pop-up will appear.

Next, provide a suitable name for your custom runtime and proceed by uploading the **config.yaml** file given above. Finally, ensure you save your changes by clicking on the save button.

### Import the Model in Inferless
Log in to your inferless account, select the workspace you want the model to be imported into and click the Add Model button.

Select the PyTorch as framework and choose **Repo(custom code)** as your model source and use the forked repo URL as the **Model URL**.

After the create model step, while setting the configuration for the model make sure to select the appropriate runtime.

Enter all the required details to Import your model. Refer [this link](https://docs.inferless.com/integrations/github-custom-code) for more information on model import.

### Add Your Hugging Face Access Token
This model requires a Hugging Face access token for authentication. You can provide the token in the following ways:

- **Via the Platform UI**: Set the `HF_TOKEN` in the **Environment Variables** section.
- **Via the CLI**: Add the `HF_TOKEN` as an environment variable.

---
## Curl Command
Following is an example of the curl command you can use to make inference. You can find the exact curl command in the Model's API page in Inferless.

```bash
curl --location '<your_inference_url>' \
          --header 'Content-Type: application/json' \
          --header 'Authorization: Bearer <your_api_key>' \
          --data '{
  "inputs": [
              {
                "data": [
                   " logo for Car modification"
                ],
                "name": "prompt",
                "shape": [
                  1
                ],
                "datatype": "BYTES"
              },
              {
                "data": [
                   "bad art, ugly, deformed, watermark, duplicated, multiple images"
                ],
                "name": "negative",
                "shape": [
                  1
                ],
                "datatype": "BYTES"
              },{
                "data": [
                   "orange, yellow, black, purple)"
                ],
                "name": "color",
                "shape": [
                  1
                ],
                "datatype": "BYTES"
              }
            ]
          }
   '
```

---
## Customizing the Code
Open the `app.py` file. This contains the main code for inference. It has three main functions, initialize, infer and finalize.

**Initialize** -  This function is executed during the cold start and is used to initialize the model. If you have any custom configurations or settings that need to be applied during the initialization, make sure to add them in this function.

**Infer** - This function is where the inference happens. The argument to this function `inputs`, is a dictionary containing all the input parameters. The keys are the same as the name given in inputs. Refer to [input](#input) for more.

```python
    def infer(self, inputs):
        """
        Generates an image based on the provided prompt.
        """
        prompt = inputs["prompt"]
        negative = inputs["negative"]
        color = inputs["color"]
        complete_prompt = f'logo, {prompt} colors ({color})'
  
        pipeline_output_image = self.pipe(
            prompt=complete_prompt,
            negative_prompt = negative,
            num_inference_steps=30,
            guidance_scale=7,
        ).images[0]

        # Encode the generated image as a base64 string for convenient transfer
        buff = BytesIO()
        pipeline_output_image.save(buff, format="PNG")
        img_str = base64.b64encode(buff.getvalue())
        return {"generated_image_base64": img_str.decode("utf-8")}
```

**Finalize** - This function is used to perform any cleanup activity for example you can unload the model from the gpu by setting `self.pipe = None`.


For more information refer to the [Inferless docs](https://docs.inferless.com/).
