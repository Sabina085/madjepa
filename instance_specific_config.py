with open("instance_type", "r") as f:
    instance_type=f.read()

if instance_type == "personal":
    # personal instance
    DATASET_PATH="/home/sabina/PhD/BMAD"
    ROOT_DIRECTORY_EXPERIMENTS_PATH = "/home/sabina/PhD/IJepa-AD/ijepa-anomaly-detection"
elif instance_type == "university":
    # university instance
    DATASET_PATH= "/media/vhondru/hdd/istoian/BMAD/dataset"
    ROOT_DIRECTORY_EXPERIMENTS_PATH = "/media/vhondru/hdd/istoian/ijepa-anomaly-detection"
else:
      raise ValueError("The instance type was not selected properly! Choose between (personal, university)")

print(DATASET_PATH)
print(ROOT_DIRECTORY_EXPERIMENTS_PATH)
