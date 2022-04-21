import splitfolders

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
