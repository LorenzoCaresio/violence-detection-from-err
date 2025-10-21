# Violence Detection from Emergency Room Reports

Code repository for the paper *Violence Detection from Emergency Room Reports*.

Citation: L. Caresio, M. Delsanto, C. J. Scozzaro, E. Mensa, D. Colla, C. Mamo, A. Pitidis, A. Vitale, D. P. Radicioni. Violence Detection from Emergency Room Reports. To appear in the *Proceedings of the IEEE International Conference on Metrology for eXtended Reality, Artificial Intelligence and Neural Engineering - IEEE MetroXRAINE 2025*.

>This paper presents a work to discriminate emergency room reports containing violent injuries from those whose injuries are caused by other factors. Real-word clinical narratives from emergency room reports are analyzed. We report the results obtained by experimenting with multiple architectures and assess their sustainability in settings with limited computational resources and time constraints. Our best models showed to be robust to medical and clinical language, and to differences in reporting practices adopted by different hospitals, exhibiting high accuracy and running times suitable for implementation in real settings: in the violence detection task our system revealed thousands of records not previously annotated as containing injuries of violent origin; in the binary categorization task our best performing models obtained $97.7\%$ F1 score; in the multiclass categorization task a $74.6\%$ average F1 score in the categorization of violence perpetrators was found. Although further efforts are necessary to enable automatic systems to actively contribute to public health monitoring and clinical intervention, the obtained results can help bridge scientific research and everyday clinical practice.

## Data disclosure

Even though the code is publicly available, the data used in this work cannot be shared due to privacy concerns (even considering the anonymization procedures applied). For this reason, we are unable to provide access to the dataset used in the paper experiments. For further enquiries, we encourage interested researchers to contact us via our institutional email addresses (to be found in the paper).

## Replication instructions

We are aware that the lack of access to the dataset hinders the possibility to replicate our experiments, nonetheless we encourage to inspect the code to verify the soundness of our methodology and of our implementation.

W.r.t. cross-validated experiments (sec. 3.A.2 and 3.A.3 of the paper), consider `run/binaryclass_experiments.py` and `run/multiclass_experiments.py` (where you can specify which architectures to use) as the main entry points. Inspect the `model` directory to check the implementation of specific architectures. `run/detector.py` contains several functions that have been used to execute several aspects that pertain to the Violence Detection task (sec. 3.A.1), such as detecting violent reports over the whole unannotated dataset, composing the gold dataset, the human validation procedure, etc. While not all of these functions have been used in the experiments reported in the paper (some have been used in previous stages of the project), we decided to include them for the sake of completeness.

Dependencies to run the code can be found in the `requirements` directory (please use different Python environments to run experiments on BERT models vs LLM, since they require conflicting dependencies, refer to `requirements/vd-bert_requirements.txt` and `requirements/vd-llm_requirements.txt` respectively).

For any further questions, please contact us directly.
