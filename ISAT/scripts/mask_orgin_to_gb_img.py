from PIL import Image
import os


def get_masks(masks_dir, output_dir, custom_suffix):
    # 定义支持的图片文件扩展名列表
    supported_extensions = ['.png', '.jpg', '.jpeg']
    os.makedirs(output_dir, exist_ok=True)
    # 遍历遮罩文件夹中的所有图像
    for filename in os.listdir(masks_dir):
        # 检查文件扩展名
        _, file_extension = os.path.splitext(filename)
        if file_extension.lower() in supported_extensions:
            mask_filepath = os.path.join(masks_dir, filename)
            output_filename = os.path.splitext(filename)[0] + custom_suffix + os.path.splitext(filename)[1]
            output_filepath = os.path.join(output_dir, output_filename)
            if os.path.isfile(mask_filepath):
                # 打开遮罩和源图像
                mask_image = Image.open(mask_filepath).convert('L')

                pixels = mask_image.load()
                for y in range(mask_image.height):  # for every row
                    for x in range(mask_image.width):  # for each column
                        if pixels[x, y] > 0:
                            pixels[x, y] = 255
                        else:
                            pixels[x, y] = 0

                # 保存处理后的图像到输出文件夹
                mask_image.save(output_filepath)
    print("处理完毕，处理过的图片已保存到: ", output_dir)


def convert_by_mask(mask_dir, src_dir, output_dir, set_color, custom_suffix):
    # 定义支持的图片文件扩展名列表
    supported_extensions = ['.png', '.jpg', '.jpeg']
    os.makedirs(output_dir, exist_ok=True)
    # 遍历遮罩文件夹中的所有图像
    for filename in os.listdir(mask_dir):
        # 检查文件扩展名
        _, file_extension = os.path.splitext(filename)
        if file_extension.lower() in supported_extensions:
            mask_filepath = os.path.join(mask_dir, filename)
            src_filepath = os.path.join(src_dir, filename)
            output_filename = os.path.splitext(filename)[0] + custom_suffix + os.path.splitext(filename)[1]
            output_filepath = os.path.join(output_dir, output_filename)
            if os.path.isfile(mask_filepath) and os.path.isfile(src_filepath):
                # 打开遮罩和源图像
                mask_image = Image.open(mask_filepath).convert('L')
                image = Image.open(src_filepath).convert('RGBA')

                pixels = mask_image.load()
                for y in range(mask_image.height):  # for every row
                    for x in range(mask_image.width):  # for each column
                        if pixels[x, y] > 0:
                            pixels[x, y] = 0
                        else:
                            pixels[x, y] = 255

                # 创建一个纯色的新图像，颜色和遮罩图像大小相同
                color_image = Image.new('RGBA', image.size, set_color)
                # 使用遮罩图像将新的纯色图像粘贴到源图像上
                image.paste(color_image, (0, 0), mask_image)
                # 保存处理后的图像到输出文件夹
                image.save(output_filepath)
    print("处理完毕，处理过的图片已保存到: ", output_dir)


if __name__ == '__main__':
    # # 待处理的遮罩图像文件夹路径
    # mask_dir = r'G:\xiaowu-pic\133_3channel\conflicts'
    # # 待处理的源图片文件夹路径
    # src = r'G:\xiaowu-pic\133_select_new'
    # # 输出的文件夹路径
    # output = r'G:\xiaowu-pic\133_3channel\conflicts'
    # # 设置的颜色值
    # color_red = (255, 0, 0, 255)  # 红色
    # color_green = (0, 255, 0, 255)
    # # 文件名末尾添加的自定义后缀
    # suffix_red = "_r"
    # suffix_green = "_g"
    #
    # # 输出一个红背景，一个绿背景
    # convert_by_mask(mask_dir, src, output, color_red, suffix_red)
    # convert_by_mask(mask_dir, src, output, color_green, suffix_green)
    pass