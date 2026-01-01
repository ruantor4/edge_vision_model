# Edge Vision Model – Modelagem e Análise de Algoritmos de Detecção de Objetos

Aplicação desenvolvida em **[Python 3.11](https://docs.python.org/pt-br/3.11/contents.html)** para realizar **modelagem, treinamento, avaliação e análise comparativa de algoritmos de detecção de objetos**, como parte de uma **Prova de Conceito (PoC)** em **Visão Computacional**.

O projeto tem como objetivo transformar os insumos produzidos pelo **Edge Vision EDA** em **modelos de detecção treinados e avaliados**, seguindo uma **pipeline profissional, reprodutível e auditável**, permitindo comparar diferentes arquiteturas e **subsidiar a escolha técnica do modelo mais adequado** para uso posterior em sistemas de inferência em **computação de borda (edge computing)**.

Este repositório é **exclusivamente dedicado à etapa de modelagem e análise de algoritmos**, não realizando **Análise Exploratória de Dados (EDA)**, **inferência em tempo real** ou **integração com sistemas finais**, as quais são tratadas em projetos correlatos do ecossistema Edge Vision.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Objetivos do Projeto

- Preparar datasets de detecção de objetos de forma controlada e rastreável.
- Treinar diferentes algoritmos de detecção de objetos.
- Avaliar o desempenho dos modelos em conjunto de teste independente.
- Analisar métricas de qualidade e trade-offs computacionais.
- Comparar arquiteturas de forma justa e reprodutível.
- Selecionar modelos candidatos para uso em ambientes de edge computing.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Estratégia de Modelagem

A estratégia adotada neste projeto segue práticas profissionais de engenharia de Machine Learning:

- **Separação clara entre EDA, modelagem e inferência**
- **Uso de dataset externo imutável**, previamente analisado no projeto Edge Vision EDA
- **Preparação controlada dos dados**, específica para cada algoritmo
- **Treinamento isolado por arquitetura**
- **Avaliação em conjunto de teste nunca utilizado durante o treinamento**
- **Comparação objetiva baseada em métricas padronizadas**

Cada algoritmo é treinado e avaliado sob **condições equivalentes**, garantindo uma comparação técnica justa.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Algoritmos Avaliados

Os seguintes algoritmos de detecção de objetos são utilizados e comparados:

- **YOLO (You Only Look Once)**  
  Arquitetura de estágio único, otimizada para inferência rápida, especialmente adequada para aplicações em edge computing.

- **SSD (Single Shot Detector)**  
  Detector de estágio único com foco em simplicidade e desempenho intermediário entre velocidade e precisão.

- **Faster R-CNN**  
  Arquitetura de dois estágios, com maior custo computacional, utilizada como referência de precisão.

Cada modelo é treinado com configurações explícitas e avaliado utilizando o mesmo conjunto de métricas.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Funcionalidades

| Categoria | Descrição |
|-----------|-----------|
| **Preparação de Dados** | Conversão e organização do dataset conforme requisitos de cada algoritmo. |
| **Treinamento de Modelos** | Treinamento controlado de YOLO, SSD e Faster R-CNN. |
| **Avaliação de Desempenho** | Cálculo de métricas em conjunto de teste isolado. |
| **Análise Comparativa** | Comparação objetiva de desempenho entre algoritmos. |
| **Versionamento de Artefatos** | Registro de dados preparados, pesos treinados e métricas por execução. |
| **Persistência de Resultados** | Salvamento estruturado de métricas e logs. |
| **Pipeline Reprodutível** | Execução determinística baseada em configurações versionadas. |

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Métricas de Avaliação

A avaliação dos modelos é realizada utilizando métricas amplamente aceitas na literatura e na indústria:

- **Precision**
- **Recall**
- **mAP@0.5**
- **mAP@0.5:0.95**
- **Tempo de inferência**

As métricas são definidas **antes do treinamento** e aplicadas de forma consistente a todos os modelos.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Estrutura de Diretórios

```bash
edge-vision-model/
├── config/
│       ├── settings.py              # Configurações centrais do projeto
│       ├── dataset.yaml             # Referência ao dataset externo
│       ├── metrics.yaml             # Definição das métricas de avaliação
│       └── models/
│           ├── yolo.yaml
│           ├── ssd.yaml
│           └── faster_rcnn.yaml
│
├── core/
│      ├── dataset_preparer.py       # Preparação controlada do dataset
│      ├── trainer.py                # Treinamento dos modelos
│      ├── evaluator.py              # Avaliação dos modelos
│      └── comparator.py             # Comparação entre algoritmos
│
├── viz/
│     └── plots.py                   # Visualizações comparativas (opcional)
│
├── artifacts/
│      ├── prepared_data/            # Datasets preparados (versionados)
│      ├── models/                   # Pesos treinados
│      ├── metrics/                  # Métricas de avaliação
│      └── comparisons/              # Resultados comparativos
│
├── utils/
│      └── logging_global.py         # Logging global do sistema
│
├── logs/
│
├── main.py                          # Orquestração da pipeline de modelagem
│
├── requirements.txt
│
└── README.md
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Dataset

O dataset utilizado neste projeto é externo ao repositório e corresponde ao dataset previamente analisado e validado no projeto Edge Vision EDA.

O acesso ao dataset é realizado exclusivamente em modo leitura, por meio de configuração definida em config/dataset.yaml.
Nenhuma modificação é realizada sobre o dataset original durante a execução deste projeto.

Os dados preparados para treinamento são gerados e versionados no diretório artifacts/prepared_data, garantindo rastreabilidade entre dados, modelos e métricas.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## Execução

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

### Passo 3 – Executar o EDA
```bash
$ python main.py
```
A execução realiza, de forma controlada, as etapas de preparação de dados, treinamento, avaliação e comparação dos modelos, conforme definido em configuração.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Observações Técnicas

- O projeto é focado exclusivamente em modelagem e análise de algoritmos.
- Não há qualquer etapa de EDA ou inferência em tempo real neste repositório.
- O conjunto de teste é mantido isolado e nunca utilizado durante o treinamento.
- Os modelos selecionados são utilizados em projetos correlatos do ecossistema Edge Vision.
- O `main.py` atua apenas como orquestrador da pipeline.s