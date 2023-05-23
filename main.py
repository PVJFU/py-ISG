import imageio as iio
import numpy as np
import os
import random
import string
import threading
from PIL import Image

# Default Settings
# set 0 to not resize (on 0 files are completely readable after the saving, but YouTube will damage it)
# Videos with higher resolution will be almost the same by size (sometimes even bigger) but they will process faster as
# there are fewer frames to process
RESIZE_TIMES = 4
WIDTH = 1280 if not RESIZE_TIMES else int(1280 / RESIZE_TIMES)
HEIGHT = 720 if not RESIZE_TIMES else int(720 / RESIZE_TIMES)
BYTES_PER_IMAGE = int(WIDTH * HEIGHT / 8)
FPS = 30

# Fork Settings
# Extra threads are only used for the video to file function. Increase if you want it to utilize more CPU and it is not already maxing it out. Decrease to use less, down to 0 to basically act single-threaded. Each thread increases RAM usage very slightly.
EXTRA_THREADS = 63
DATA_CHUNKS = [None] * (EXTRA_THREADS + 1)


def create_picture_from_bytes(byte_data):
    if BYTES_PER_IMAGE != len(byte_data):  # Surely this should only happen when at the end of a file. Clueless
        byte_data += b"\x00" * (BYTES_PER_IMAGE - len(byte_data))
    image = Image.frombytes('1', (WIDTH, HEIGHT), byte_data)
    if RESIZE_TIMES:
        image = image.resize((WIDTH * RESIZE_TIMES, HEIGHT * RESIZE_TIMES))
    image = image.convert("L")  # So iio writer will not output a black frame.
    return image


def create_bytes_from_picture(frame, index, threshold=128):
    image = Image.fromarray(frame)
    # Grayscale
    image = image.convert("L")
    # Threshold (avoid read mistakes on image compression), 128 is mid-gray pixel
    image = image.point(lambda p: 255 if p > threshold else 0)
    # To mono (1 bit per pixel image)
    image = image.convert('1')
    if RESIZE_TIMES:
        width, height = image.size
        image = image.resize((int(width / RESIZE_TIMES), int(height / RESIZE_TIMES)))
    # Convert pixels to bytes
    image = image.tobytes()
    DATA_CHUNKS[index] = image


def generate_random_file_name(size=12):
    return f"output_{''.join(random.choice(string.ascii_letters) for _ in range(size))}"


def generate_unique_file_name(size=12):
    file_name = generate_random_file_name(size)
    while os.path.exists(file_name):
        file_name = generate_random_file_name(size)
    return file_name


def file_data_chunk_generator(file):
    while True:
        chunk = file.read(BYTES_PER_IMAGE)
        if not chunk:
            break
        yield chunk


def convert_file_to_video(file_path_to_conv, output_path):
    with open(os.path.normpath(file_path_to_conv), "rb") as file:
        with iio.get_writer(f"{os.path.normpath(output_path)}.mp4", fps=FPS) as writer:  # This seems ugly. There has to be a better way to do this.
            for index, chunk in enumerate(file_data_chunk_generator(file)):  # enumerate is lazy, so it's ok to use it with generator
                image = create_picture_from_bytes(chunk)
                writer.append_data(np.asarray(image, dtype="uint8"))


def convert_video_to_file(video_path_to_conv, output_path):
    with open(f"{output_path}.zip", "ab") as archive:
        threads = []
        data_chunk_index = 0
        for index, frame in enumerate(iio.imiter(os.path.normpath(video_path_to_conv))):
            thread = threading.Thread(target=create_bytes_from_picture, args=(frame, data_chunk_index))
            threads.append(thread)
            thread.start()
            if data_chunk_index != EXTRA_THREADS:
                data_chunk_index += 1
            else:
                for thread in threads:
                    thread.join()
                threads.clear()
                for data_chunk in DATA_CHUNKS:
                    archive.write(data_chunk)
                data_chunk_index = 0
        DATA_CHUNKS[data_chunk_index] = None
        for thread in threads:
            thread.join()
        for data_chunk in DATA_CHUNKS:
            if data_chunk == None:
                break
            archive.write(data_chunk)


def make_convertion(convert_func):
    convert_object = "file" if convert_func == convert_file_to_video else "video"
    path_file_to_convert = input(f"Input path of the {convert_object} you want to convert: ")
    print("\nIn progress. Please wait...")
    output_file_name = generate_unique_file_name()
    convert_func(path_file_to_convert, output_file_name)
    # get file name with extension
    current_dir = os.path.dirname(os.path.realpath(__file__))
    created_file_full_name = next((ext for ext in os.listdir(current_dir) if output_file_name in ext))
    print(f"\nCreated file: {created_file_full_name}")


def main():
    while True:
        convert_option = input("Choose an option:\n1. FILE -> VIDEO\n"
                               "2. VIDEO -> FILE\nYour choice: ")
        # input validation
        if convert_option not in ("1", "2"):
            continue
        # FILE -> VIDEO
        if convert_option == "1":
            make_convertion(convert_file_to_video)
        # VIDEO -> FILE
        if convert_option == "2":
            make_convertion(convert_video_to_file)


if __name__ == "__main__":
    main()
