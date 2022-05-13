import splitfolders

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

splitfolders.ratio(
    "C:/Users/ant_on/Desktop/data_mask/",
    output="./data",
    seed=42,
    ratio=(0.7, 0.2, 0.1),
    group_prefix=None,
)  # default values
