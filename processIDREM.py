import numpy as np
import os
import threading
import subprocess
import shutil
from pathlib import Path


def getClusterPaths(edges, total_stages):
    '''
    Obtain the paths of each cluster for multiple stages.

    parameters
    -----------
    edges: list
        A list of lists, where each sublist contains edges between consecutive stages.
    total_stages: int
        Total number of stages.

    return
    -----------
    paths: list
        A collection of paths of clusters.
    '''
    if len(edges) != total_stages - 1:
        raise ValueError("Number of edges must be one less than total stages")

    if isinstance(edges, dict):
        edges = {int(key): value for key, value in edges.items()}
    else:
        edges = {index: value for index, value in enumerate(edges)}
    paths = {}
    # Initialize paths with the first set of edges
    for each in edges[0]:
        if str(each[0]) not in paths:
            paths[str(each[0])] = [[each[0]], [each[1]]]
        else:
            paths[str(each[0])][1].append(each[1])

    # Iterate through remaining stages
    for stage in range(1, total_stages - 1):
        for each in edges[stage]:
            for item in paths.keys():
                if len(paths[item]) <= stage:
                    continue
                if each[0] in paths[item][stage]:
                    if len(paths[item]) == stage + 1:
                        paths[item].append([each[1]])
                    else:
                        paths[item][stage + 1].append(each[1])

    return paths


def getClusterIdrem(paths, state, total_stages):
    '''
    Concatenate the average gene expression in a cluster tree. Shape: [number of stages, number of genes]

    parameters
    -----------
    paths: The collection of paths.
    state: A list of average gene expression of each state.
    total_stages: Total number of stages.

    return
    -----------
    out: A list of gene expression of each cluster tree.
    '''
    out = []

    for path_key in paths.keys():
        path = paths[path_key]

        # Ensure the path contains the expected number of stages
        if len(path) == total_stages:
            stages = [averageNode(node, state[i]) for i, node in enumerate(path)]

            # Reshape each stage and concatenate
            reshaped_stages = [stage.reshape(-1, 1) for stage in stages]
            joint_matrix = np.concatenate(reshaped_stages, axis=1)

            out.append(joint_matrix)

    return out


def getIdrem(paths, state):
    '''
    concatenate the average gene expression of clusters in each path. shape: [number of stages, number of gene]
    parameters
    ----------------------
    paths: list
        the list of paths
    state: list
        a list of average gene expression of each state

    return
    ----------------------
    out: list
        a list of gene expression of each path
    '''
    out = []
    for path in paths:
        stages = [state[stage][nodes].mean(axis=0).reshape(-1, 1)
                  for stage, nodes in enumerate(path)]
        out.append(np.concatenate(stages, axis=1))
    return out


class IDREMthread(threading.Thread):
    '''
    the thread for running IDREM. Support multiple threads.
    '''

    def __init__(self, indir, outdir, each, idrem_dir):
        threading.Thread.__init__(self)
        self.indir = indir
        self.outdir = outdir
        self.each = each
        self.idrem_dir = idrem_dir
        self.error = None

    def run(self):
        try:
            idrem_dir = Path(self.idrem_dir)
            jar_path = idrem_dir / "idrem.jar"
            if not jar_path.is_file():
                raise FileNotFoundError(f"iDREM JAR not found: {jar_path}")
            Path(self.outdir).mkdir(parents=True, exist_ok=True)
            command = ["java", "-Xmx8G", "-jar", str(jar_path), "-b", str(self.indir), str(self.outdir)]
            result = subprocess.run(command, cwd=idrem_dir, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"iDREM failed for {self.indir}:\n{result.stdout}\n{result.stderr}"
                )
        except Exception as exc:
            self.error = exc


def runIdrem(paths, midpath, idremInput, genenames, iteration, idrem_dir, species='Human',
             Minimum_Standard_Deviation=0.01, Convergence_Likelihood=0.1, Minimum_Absolute_Log_Ratio_Expression=0.05,
             trained=False):
    '''
    train IDREM model and save the results in iterative training with midpath and iteration

    parameters
    ----------------------
    paths: the path of IPF progression
    idremInput: average gene expression of each path
    trained: if the model is trained, use saved model

    '''
    iteration_dir = Path(midpath) / str(iteration)
    dir1 = iteration_dir / "idremInput"
    dir2 = iteration_dir / "idremsetting"
    dir3 = iteration_dir / "idremModel"
    for directory in (dir1, dir2, dir3):
        directory.mkdir(parents=True, exist_ok=True)

    idrem_root = Path(idrem_dir)
    examplefile_path = idrem_root / "example_settings.txt"
    if not examplefile_path.is_file():
        raise FileNotFoundError(f"iDREM settings template not found: {examplefile_path}")
    if species not in {"Human", "Mouse"}:
        raise ValueError("species must be 'Human' or 'Mouse'.")
    reference_files = {
        "Human": ["human_encode.txt.gz", "TFInput/human_encode.txt.gz", "goa_human.gaf.gz"],
        "Mouse": ["mouse_predicted.txt.gz", "TFInput/mouse_predicted.txt.gz", "goa_mouse.gaf.gz"],
    }
    missing_references = [
        str(idrem_root / relative_path)
        for relative_path in reference_files[species]
        if not (idrem_root / relative_path).is_file()
    ]
    if missing_references:
        raise FileNotFoundError(
            "Missing iDREM reference files:\n" + "\n".join(missing_references)
        )

    settinglist = []
    for i, each in enumerate(paths):
        each_processed = []
        for e in each:
            e = str(e).strip('[]')
            e = e.replace(', ', 'n')
            each_processed.append(e)
        # each_processed = [str(e).strip('[]').replace(', ', 'n') for e in each]
        print(each_processed)
        file_name = '-'.join(each_processed)
        file_path = dir1 / f'{file_name}.txt'
        header = ['gene'] + [f'stage{j}' for j in range(len(each))]
        with file_path.open('w') as f:
            f.write('\t'.join(header) + '\n')
            for j, row in enumerate(idremInput[i]):
                row_data = '\t'.join(str(r) for r in row)
                f.write("%s\t%s\n" % (genenames[j], row_data))
        settings_file_path = dir2 / f'{file_name}.txt'
        with examplefile_path.open('r') as examplefile:
            with settings_file_path.open('w') as f:
                for k, line in enumerate(examplefile.readlines()):

                    if trained and k == 4:
                        print(midpath)
                        f.write('%s\t%s\n' % ('Saved_Model_File', os.path.join(
                            dir1.resolve(), f'{file_name}.txt')))
                    elif k == 1:
                        if species == 'Human':
                            f.write('%s\t%s\n' % ('TF-gene_Interaction_Source', 'human_encode.txt.gz'))

                            continue
                        elif species == 'Mouse':
                            f.write('%s\t%s\n' % ('TF-gene_Interaction_Source', 'mouse_predicted.txt.gz'))
                            continue
                    elif k == 2:
                        if species == 'Human':
                            f.write('%s\t%s\n' % ('TF-gene_Interactions_File', 'TFInput/human_encode.txt.gz'))
                            continue
                        elif species == 'Mouse':
                            f.write('%s\t%s\n' % ('TF-gene_Interactions_File', 'TFInput/mouse_predicted.txt.gz'))
                            continue
                    elif k == 5:
                        if species == 'Human':
                            f.write('%s\t%s\n' % ('Gene_Annotation_Source', 'Human (EBI)'))
                            continue
                        elif species == 'Mouse':
                            f.write('%s\t%s\n' % ('Gene_Annotation_Source', 'Mouse (EBI)'))
                            continue
                    elif k == 6:
                        if species == 'Human':
                            f.write('%s\t%s\n' % ('Gene_Annotation_File', 'goa_human.gaf.gz'))
                            continue
                        elif species == 'Mouse':
                            f.write('%s\t%s\n' % ('Gene_Annotation_File', 'goa_mouse.gaf.gz'))
                            continue
                    elif k == 17:
                        f.write('%s\n' % ('miRNA-gene_Interaction_Source'))
                        continue
                    elif k == 18:
                        f.write('%s\n' % ('miRNA_Expression_Data_File'))
                        continue
                    elif k == 26:
                        f.write('%s\n' % ('Proteomics_File'))
                        continue
                    elif k == 34:
                        f.write('%s\n' % ('Epigenomic_File'))
                        continue
                    elif k == 35:
                        f.write('%s\n' % ('GTF File'))
                        continue
                    elif k == 42:
                        f.write('%s\t%s\n' % (
                        'Minimum_Absolute_Log_Ratio_Expression', str(Minimum_Absolute_Log_Ratio_Expression)))
                        continue
                    elif k == 51:
                        f.write('%s\t%s\n' % ('Convergence_Likelihood_%', str(Convergence_Likelihood)))
                        continue
                    elif k == 52:
                        f.write('%s\t%s\n' % ('Minimum_Standard_Deviation', str(Minimum_Standard_Deviation)))
                        continue
                    elif k != 3:
                        f.write(line)
                    else:
                        f.write('%s\t%s\n' % ('Expression_Data_File', os.path.join(
                            dir1.resolve(), f'{file_name}.txt')))

        settinglist = os.listdir(dir2)

    print("settinglist is:" ,settinglist)
    threads = []
    for each in settinglist:
        if each[0] != '.':
            indir = str((dir2 / each).resolve())
            outdir = str((dir3 / each).resolve())
            threads.append(IDREMthread(indir, outdir, each, idrem_root))
    count = 0
    while True:
        if count < len(threads) and count + 2 > len(threads):
            threads[count].start()
            threads[count].join()
            break
        elif count == len(threads):
            break
        else:
            threads[count].start()
            threads[count + 1].start()
            threads[count].join()
            threads[count + 1].join()
            count += 2
    errors = [thread.error for thread in threads if thread.error is not None]
    if errors:
        raise RuntimeError("One or more iDREM jobs failed.") from errors[0]
    if not trained:
        print(os.getcwd())
        results_dir = iteration_dir / "idremResults"
        if results_dir.exists():
            shutil.rmtree(results_dir)
        results_dir.mkdir(parents=True)
        for visualization in dir1.glob("*.txt_viz"):
            shutil.move(str(visualization), str(results_dir / visualization.name))
    print('idrem Done')


def averageNode(nodes, state):
    '''
    calculate the average gene expression of sibling nodes

    parameters
    ----------------------
    nodes: int
        number of sibling nodes
    state: list
        the gene expression of each cluster in a certain stage

    return
    -----------
    out: the average gene expression of sibling nodes
    '''
    out = 0
    for each in nodes:
        out += state[each]
    return out / len(nodes)


