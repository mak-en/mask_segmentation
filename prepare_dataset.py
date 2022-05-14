import splitfolders
import argparse

# Split folders with files (e.g. images) into train, validation and
# test (dataset) folders.
#
# The input folder should have the following format:
#
# input/
#     class1/
#         img1.jpg
#         img2.jpg
#         ...
#     class2/
#         imgWhatever.jpg
#         ...
#     ...
# In order to give you this:
#
# output/
#     train/
#         class1/
#             img1.jpg
#             ...
#         class2/
#             imga.jpg
#             ...
#     val/
#         class1/
#             img2.jpg
#             ...
#         class2/
#             imgb.jpg
#             ...
#     test/
#         class1/
#             img3.jpg
#             ...
#         class2/
#             imgc.jpg
#             ...

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`,
# i.e, `(.8, .2)`.
# Train, val, test

# Parser for the script arguments
parser = argparse.ArgumentParser(description="Dataset formation")
parser.add_argument(
    "--in_data_path", type=str, help="path to the original data folder"
)
parser.add_argument(
    "--out_data_path", type=str, help="path to the output data folder"
)
args = parser.parse_args()

splitfolders.ratio(
    args.in_data_path,
    output=args.out_data_path,
    seed=42,
    ratio=(0.7, 0.2, 0.1),
    group_prefix=None,
)  # default values
