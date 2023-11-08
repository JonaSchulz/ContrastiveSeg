from argparse import ArgumentParser
import os


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", default=None, type=str, dest="name")
    parser.add_argument("--template", default="HRNet48_Contrastive", type=str, dest="template")
    args_parser = parser.parse_args()

    name = args_parser.name
    template_name = args_parser.template

    with open(f"../configs/thesis/{template_name}.json") as config_template:
        config_template = config_template.read()
        config_template = config_template.replace(template_name, name)

    with open(f"../condor/{template_name}/run_{template_name}.tbi") as tbi_template:
        tbi_template = tbi_template.read()
        tbi_template = tbi_template.replace(template_name, name)

    with open(f"../condor/{template_name}/run_{template_name}.sh") as sh_template:
        sh_template = sh_template.read()
        sh_template = sh_template.replace(template_name, name)

    os.makedirs(f"../condor/{name}", exist_ok=True)

    with open(f"../configs/thesis/{name}.json", "w") as config:
        config.write(config_template)

    with open(f"../condor/{name}/run_{name}.tbi", "w") as tbi:
        tbi.write(tbi_template)

    with open(f"../condor/{name}/run_{name}.sh", "w") as sh:
        sh.write(sh_template)

