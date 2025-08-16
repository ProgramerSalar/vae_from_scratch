## Prepare the Dataset

The training dataset should be arranged into a json file, with `video`, `text` fields. Since the video vae latent extraction is very slow, we strongly recommend you to pre-extract the video vae latents to save the training time. We provide a video vae latent extraction script in folder `tools`. You can run it with the following command:

```bash
sh scripts/extract_vae_latent.sh
```

(optional) Since the T5 text encoder will cost a lot of GPU memory, pre-extract the text features will save the training memory. We also provide a text feature extraction script in folder `tools`. You can run it with the following command:

```bash
sh scripts/extract_text_feature.sh
```

The final training annotation json file should look like the following format:

```
{"video": video_path, "text": text prompt, "latent": extracted video vae latent, "text_fea": extracted text feature}
```

We provide the example json annotation files for [video](https://github.com/jy0205/Pyramid-Flow/blob/main/annotation/video_text.jsonl) and [image](https://github.com/jy0205/Pyramid-Flow/blob/main/annotation/image_text.jsonl)) training in the `annotation` folder. You can refer them to prepare your training dataset.