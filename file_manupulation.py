import yaml

if __name__ == '__main__':

    with open("environment.yml") as f:
        env = yaml.safe_load(f)
    with open('requirements.txt','w') as f:
        for dep in env["dependencies"]:
            if isinstance(dep, dict) and "pip" in dep:
                for ii,pkg in enumerate(dep["pip"]):
                    print(pkg)
                    if ii > 0: f.write(f'\n{pkg}')
                    else: f.write(f'{pkg}')