# Edge Vision Model – Modelagem e Análise de Algoritmos de Detecção de Objetos

Aplicação desenvolvida em **[Python 3.11](https://docs.python.org/pt-br/3.11/contents.html)** para realizar **modelagem, treinamento, avaliação e análise comparativa de algoritmos de detecção de objetos**, como parte de uma **Prova de Conceito (PoC)** em **Visão Computacional**.

O projeto tem como objetivo transformar os insumos produzidos pelo **Edge Vision EDA** em **modelos de detecção treinados e avaliados**, seguindo uma **pipeline profissional, reprodutível e auditável**, permitindo comparar diferentes arquiteturas e **subsidiar a escolha técnica do modelo mais adequado** para uso posterior em sistemas de inferência em **computação de borda (edge computing)**.

Este repositório é **exclusivamente dedicado à etapa de modelagem e análise de algoritmos**, não realizando **Análise Exploratória de Dados (EDA)**, **inferência em tempo real** ou **integração com sistemas finais**, as quais são tratadas em projetos correlatos do ecossistema Edge Vision.

O projeto foi desenvolvido com foco em clareza arquitetural, separação de responsabilidades e rastreabilidade experimental.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Objetivos do Projeto

- Preparar datasets de detecção de objetos de forma controlada e rastreável.
- Treinar diferentes algoritmos de detecção de objetos.
- Avaliar o desempenho dos modelos em conjunto de teste independente.
- Analisar métricas de qualidade e trade-offs computacionais.
- Comparar arquiteturas de forma justa e reprodutível.
- Selecionar modelos candidatos para uso em ambientes de edge computing.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Estratégia de Modelagem

A estratégia adotada neste projeto segue práticas profissionais de engenharia de Machine Learning:

- **Separação clara entre EDA, modelagem e inferência**
- **Uso de dataset externo imutável**, previamente analisado no projeto Edge Vision EDA
- **Preparação controlada dos dados**, específica para cada algoritmo
- **Treinamento isolado por arquitetura**
- **Avaliação em conjunto de teste nunca utilizado durante o treinamento**
- **Comparação objetiva baseada em métricas padronizadas**

Cada algoritmo é treinado e avaliado sob **condições equivalentes**, garantindo uma comparação técnica justa.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Algoritmos Avaliados

Os seguintes algoritmos de detecção de objetos são utilizados e comparados:

- **[YOLO](https://docs.ultralytics.com/pt/)**  
  Arquitetura de estágio único, otimizada para inferência rápida, especialmente adequada para aplicações em edge computing.

- **[SSD](https://pytorch.org/hub/nvidia_deeplearningexamples_ssd/)**  
  Detector de estágio único com foco em simplicidade e desempenho intermediário entre velocidade e precisão.

- **[Faster R-CNN](https://docs.pytorch.org/vision/master/models/faster_rcnn.html)**  
  Arquitetura de dois estágios, com maior custo computacional, utilizada como referência de precisão.

Cada modelo é treinado com configurações explícitas e avaliado utilizando o mesmo conjunto de métricas.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Funcionalidades

| Categoria | Descrição |
|-----------|-----------|
| **Preparação de Dados** | Conversão e organização do dataset conforme requisitos de cada algoritmo. |
| **Treinamento de Modelos** | Treinamento controlado de YOLO, SSD e Faster R-CNN. |
| **Avaliação de Desempenho** | Cálculo de métricas padronizadas (COCO-style) em conjunto de teste isolado. |
| **Análise Comparativa** | Comparação objetiva de desempenho entre algoritmos. |
| **Versionamento de Artefatos** | Registro de dados preparados, pesos treinados e métricas por execução. |
| **Persistência de Resultados** | Salvamento estruturado de métricas e logs. |
| **Pipeline Reprodutível** | Execução determinística baseada em configurações versionadas. |

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## Métricas de Avaliação

A avaliação dos modelos é realizada utilizando métricas padronizadas no estilo COCO, amplamente aceitas na literatura e na indústria:

- **Average Precision (AP@0.5:0.95)** – métrica principal de desempenho global.
- **Average Precision (AP@0.5)** – desempenho em limiar fixo de IoU.
- **Average Recall (AR@0.5:0.95)** – capacidade média de recuperação dos objetos.
- **Tempo médio de inferência por imagem**
- **Frames por segundo (FPS)**
- **Pico de uso de memória de GPU (VRAM)**

As métricas são definidas previamente e aplicadas de forma consistente a todos os modelos avaliados.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Tecnologias Utilizadas

| Categoria | Tecnologia |
|----------|------------|
| **Linguagem** | **[Python 3.11](https://docs.python.org/pt-br/3.11/contents.html)** |
| **Deep Learning** | **[YOLO](https://yolo-docs.readthedocs.io/en/latest/)**, **[SSD](https://docs.pytorch.org/vision/main/models/ssd.html)**, **[Faster R-CNN](https://docs.pytorch.org/vision/main/models/faster_rcnn.html)** |
| **Análise Numérica** | **[NumPy](https://numpy.org/doc/)** |
| **Avaliação** | **[pycocotools](https://github.com/cocodataset/cocoapi)**, **[métricas COCO](https://cocodataset.org/)** |
| **Visualização** | **[Matplotlib](https://matplotlib.org/)** |
| **Logging** | **[logging](https://docs.python.org/pt-br/3/library/logging.html)** |
| **Sistema** | **[pathlib](https://docs.python.org/3/library/pathlib.html)**, **[yaml](https://pyyaml.org/wiki/PyYAMLDocumentation)**, **[json](https://docs.python.org/3/library/json.html)** |

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Estrutura de Diretórios

```bash
edge-vision-model/
├── config/                     # Configurações globais e específicas dos modelos
│   ├── settings.py             # Paths, flags da pipeline e parâmetros gerais
│   ├── dataset.yaml            # Referência ao dataset externo (RAW)
│   ├── metrics.yaml            # Definição formal das métricas de avaliação
│   └── models/                 # Configurações específicas por arquitetura
│
├── core/                       # Núcleo da pipeline de modelagem
│   ├── dataset_preparer.py     # Preparação determinística do dataset por modelo
│   ├── config_validator.py     # Validação estrutural das configurações
│   └── evaluation_runner.py    # Orquestra a execução das avaliações
│
├── trainers/                   # Treinamento dos modelos
│   ├── yolo_trainer.py
│   ├── ssd_trainer.py
│   └── faster_rcnn_trainer.py
│
├── evaluators/                 # Avaliação COCO-style dos modelos
│   ├── base_coco_evaluator.py
│   ├── yolo_evaluator.py
│   ├── ssd_evaluator.py
│   └── faster_rcnn_evaluator.py
│
├── comparator/                       # Comparação entre arquiteturas
│   ├── model_comparator_metrics.py   # Comparação de métricas de qualidade
│   ├── model_comparator_cost.py      # Avaliação de custo computacional
│   └── model_comparator_merge.py     # Consolidação final dos resultados
│
├── viz/                        # Visualizações analíticas
│   └── plots.py                # Geração de gráficos comparativos
│
├── artifacts/                  # Artefatos gerados pela pipeline
│   ├── prepared_data/          # Datasets preparados
│   ├── models/                 # Pesos treinados
│   ├── metrics/                # Métricas de avaliação
│   └── comparisons/            # Resultados comparativos e consolidados
│
├── utils/                      # Utilitários compartilhados
│   └── logging_global.py       # Configuração global de logging
│
├── logs/                       # Logs de execução da pipeline
│
├── run_prepare.py              # Execução da preparação do dataset
├── run_model.py                # Execução do treinamento dos modelos
├── run_evaluate.py             # Execução da avaliação dos modelos
│
├── requirements.txt            # Dependências do projeto
└── README.md                   # Documentação técnica

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## Dataset

O dataset utilizado neste projeto é externo ao repositório e corresponde ao dataset previamente analisado e validado no projeto Edge Vision EDA.

O acesso ao dataset é realizado exclusivamente em modo leitura, por meio de configuração definida em config/dataset.yaml.
Nenhuma modificação é realizada sobre o dataset original durante a execução deste projeto.

Os dados preparados para treinamento são gerados e versionados no diretório artifacts/prepared_data, garantindo rastreabilidade entre dados, modelos e métricas.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## Execução da pipeline

A execução do Edge Vision Model é organizada em etapas explícitas e independentes,
permitindo controle total sobre cada fase do processo.

### Passo 1 – Criar ambiente virtual

```bash
$ python -m venv .venv
$ source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
```

### Passo 2 – Instalar dependências

```bash
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

### Passo 3 – Preparar dataset

Executa a preparação e organização dos dados para cada arquitetura, sem modificar
o dataset original.

```bash
$ python run_prepare.py
```
#### Passo 4 – Treinar os modelos

O treinamento é executado por modelo, de forma isolada.

```bash
$ python run_model.py train --model yolo
$ python run_model.py train --model ssd
$ python run_model.py train --model faster_rcnn
```

#### Passo 5 – Avaliar os modelos

Executa a avaliação padronizada (COCO-style) dos modelos treinados.

```bash
$ python run_evaluate.py
```

#### Passo 6 – Comparação e visualização

A comparação de métricas, custo computacional e geração de gráficos é realizada a
partir dos artefatos gerados nas etapas anteriores.

```bash
$ python -m viz.plots
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## Observações Técnicas

- O projeto é focado exclusivamente nas etapas de modelagem, avaliação e análise comparativa de algoritmos.
- Não há qualquer etapa de EDA ou inferência em tempo real neste repositório.
- O conjunto de teste é mantido isolado e nunca utilizado durante o treinamento.
- Os modelos selecionados são utilizados em projetos correlatos do ecossistema Edge Vision.
- A execução da pipeline é organizada em etapas explícitas e independentes, por meio dos scripts 
  `run_prepare.py`, `run_model.py` e `run_evaluate.py`, assegurando controle operacional, rastreabilidade 
  e reprodutibilidade experimental.