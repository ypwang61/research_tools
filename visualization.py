import numpy as np
import os
import matplotlib.pyplot as plt




class ImageCaptionVisualizer:
    def __init__(
        self,
        save_path: str,
        
        urls: np.ndarray,
        captions: np.ndarray,
        add_values: dict = None,
        
        show_array: tuple = (3, 5),
        
    ):
        """
        Initialize the ImageCaptionVisualizer, and save the urls, captions, add_values, etc.

        Args:
            save_path (str): path to save the dic of visualization results
            urls (np.ndarray):  list of urls of the images
            captions (np.ndarray): list of captions of the images
            add_values (dict, optional): dict of values that need to show together, e.g., clip_score, vass, etc. Defaults to None.
            show_array (tuple, optional): the shape of the visualization figure. Defaults to (3, 5).
        """
        self.save_path = os.path.join(save_path, 'vis')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            print(f'make dir {self.save_path}')
        else:
            print(f'{self.save_path} exists')
        
        
        self.urls = urls
        self.captions = captions
        self.add_values = add_values
        self.show_array = show_array
    
    
    def vis_datasets(
        self,
        key: str, 
        note: str = '',
        ratio_list: list = [1.0, 0.99, 0.98, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02]
    ):  
        """
        vis the images and captions, for some given ratios of the add_values (particular key)
        will show the images and captions and add_values with the add_values[key] in the top ratio.

        """
        print(f'vis_datasets: note = {note}')
        
        for ratio in ratio_list:
            self.vis_images_captions_given_add_values(ratio = ratio, add_values_key=key, note=note)
    
    def vis_images_captions_given_add_values(
        self,
        ratio: float, # 0.0~1.0
        add_values_key: str, # support clip_score, add_values
        note: str = ''
    ):
        if self.add_values is None:
            index = np.arange(len(self.urls))
        else:
            add_values = self.add_values[add_values_key]
            if ratio == 1.0:
                # select the top 100 values
                neg_start = max(0, len(add_values)-100)
                index = np.argsort(add_values)[neg_start:]
            else:
                target_add_value_upper = np.quantile(add_values, min(ratio+0.01,1.0), interpolation='higher')
                target_add_value_lower = np.quantile(add_values, max(ratio-0.01,0.0))
                # target_clipscore +- 1% ratio
                index = np.where((add_values>=target_add_value_lower) \
                                    & (add_values<=target_add_value_upper))[0]
        
        print(f'len of index: {len(index)}')
        
        urls_selected = self.urls[index]
        captions_selected = self.captions[index]
        clipscores_selected = self.clipscores[index]
        add_values_selected = {}
        for key, value in self.add_values.items():
            add_values_selected[key] = value[index]
        
        
        self.vis_images_captions(
            note=note + f'_{add_values_key}_ratio_{ratio:.2f}',
            urls=urls_selected,
            captions=captions_selected,
            clipscores=clipscores_selected,
            add_values=add_values_selected,

        )
        
    
    
    def print_error(self, f, url, caption, add_value, reason='fail to download image', print_error = False):
        if not print_error:
            return
        f.write(f'=========================================================')
        f.write(f'error: {reason}')
        
        l = f'url: {url}\ncaption: {caption}\n'
        for key, value in add_value.items():
            l += f'\n{key}: {value}'
        f.write(l)
        f.write(f'=========================================================')
    
    
    
    def vis_images_captions(
        self,
        note: str = '',
        
        urls: np.ndarray = None,
        captions: np.ndarray = None,
        add_values: dict = None,
        
    ):
        """
        vis the images and captions
        
        Args:
            note (str, optional): note for visualization. Defaults to ''.
            urls (np.ndarray, optional):  list of urls of the images. Defaults to None.
            captions (np.ndarray, optional): list of captions of the images. Defaults to None.
            add_values (dict, optional): dict of values that need to show together, e.g., clip_score, vass, etc. Defaults to None.
        """
        if urls is None:
            urls = self.urls
        if captions is None:
            captions = self.captions
            
        if add_values is None:
            add_values = self.add_values
        
        
        # obtain the images and captions
        import requests
        from PIL import Image
        from io import BytesIO
        
        # get the first num_show and last num_show images and captions with clipscores
        # show the first num_show in the first row, and the last num_show in the second row
        # show the image, caption and clipscore in each subplot, then save the figure
        
        # support chinese font
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axs = plt.subplots(self.show_array[0], self.show_array[1], figsize=(20, 15))
        
        
        # print the captions into a file
        caption_file = os.path.join(self.save_path,  f'captions_{note}.txt')
        print(f'print captions to {caption_file}')
        
        with open(caption_file, 'w') as f:
            
            index = -1
            
            i, j = 0, 0
            success_flag = False
            while 1:
                    index += 1 # index increase
                    if success_flag: 
                        # last time success, so i and j should be updated
                        i +=1
                        if i == self.show_array[0]:
                            i = 0
                            j += 1
                            if j == self.show_array[1]:
                                break
                    
                    success_flag = False
                    # index = i*self.show_array[1]+j
                    url = urls[index]
                    caption = captions[index]
                    add_value = {}
                    if add_values is not None:
                        for key, value in add_values.items():
                            add_value[key] = value[index]
                    
                    try:
                        request = requests.get(url, stream=True, )
                        
                        if request.status_code == 200: # success
                            # print(f'{index}: caption: {caption}')
                            
                            try:
                                image = Image.open(BytesIO(request.content))
                                f.write(f'({i}, {j}): \nurl: {url}\ncaption: {caption}\n')
                            except:
                                # self.print_error(f, url, caption, add_value, reason='fail to open image')
                                continue
                                                    
                            axs[i, j].imshow(image)
                            axs[i, j].axis('off')
                            
                            title = ''
                            for key, value in add_value.items():
                                title += f'{key}: {value:.4f}\n'
                            
                            # add caption to the title, but restrict the range of the caption so that it will not influence other subplots
                            # every 30 characters, add a '\n'. The total length of caption should be less than 200
                            caption = caption.replace('\n', ' ')
                            caption = caption.replace('\r', ' ')
                            caption = caption.replace('\t', ' ')
                            caption = caption[:200]
                            for k in range(0, len(caption), 30):
                                end = min(k+30, len(caption))
                                title += '\n' + caption[k:end]
                                
                            axs[i, j].set_title(title)
                            
                            success_flag = True
                        else:
                            # self.print_error(f, url, caption, add_value, reason='fail to download image')
                            pass
                    except requests.exceptions.RequestException as e:
                        # self.print_error(f, url, caption, clip_score, add_value, reason=str(e))
                        pass
                        
                
        pic_name = os.path.join(self.save_path, f'images_{note}.png')
                
        plt.savefig(pic_name)
        print(f'save image to {pic_name}, txt in {caption_file}')
        
        
        
        

