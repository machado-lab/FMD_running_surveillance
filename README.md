# Pipeline de Vigilância de Risco de FA

Pontuação de risco de Febre Aftosa (FA) em nível de propriedade rural para o Rio Grande do Sul (RS), Brasil.  
O pipeline integra dados espaciais, de rede, ambientais e de biossegurança para produzir  
dois escores compostos de risco por propriedade: peso igual e peso epidemiologicamente ponderado.

---

> [!WARNING]
> **Os dados em `raw_data/` são totalmente sintéticos (fictícios).**
> Coordenadas GPS, nomes, CPF/CNPJ e identificadores de propriedade foram
> substituídos por valores aleatórios apenas para demonstrar o funcionamento do
> código — **não representam dados reais de vigilância** e não permitem
> identificação individual. Dados públicos (estradas, fronteiras, limites do RS,
> temperatura/LST, levantamento de javalis por município) são reais.
> Para regenerar os dados sintéticos: `python tools/generate_mock_raw_data.py`.


## Estrutura do projeto

```
project_root/
├── pipeline/               # 10 scripts sequenciais de análise
│   ├── 1_load_data.py
│   ├── 2_1_javalis_spatial_distribution.py
│   ├── 2_get_temperature_rs_with_raster.py
│   ├── 3_1_road_animal_preassure.py
│   ├── 3_get_spatial_net_features.py
│   ├── 4_load_and_check_sisbravet_data.py
│   ├── 5_distance_to_border.py
│   ├── 6_final_score_cadastro_with_sisbravet_labels.py
│   ├── 7_combine_and_check_data.py
│   └── 8_fmd_risk_analysis.py
│
├── raw_data/               # Dados de entrada (somente leitura, não versionados)
│   ├── cadastro/           # CSVs de cadastro das propriedades
│   ├── gtas/               # Registros de movimentação animal (por espécie)
│   │   ├── bovine_movs/
│   │   ├── bubaline_movs/
│   │   ├── ovi_movs/
│   │   ├── capri_movs/
│   │   └── suine_movs/
│   ├── javali_data/        # Arquivo Excel do levantamento de javalis
│   ├── roads/              # roads_rs_major.shp
│   ├── sisbravet/          # Arquivos Excel de vigilância do SISBRAVET
│   ├── PNEFA_epicollec_bio_records/  # form-1__form-pvbr.csv
│   ├── rs_boundary_ibge/   # RS_UF_2024.shp (opcional, para recorte de fronteira)
│   ├── border/             # Linhas de fronteira Natural Earth (download automático)
│   └── hydro/              # Rios Natural Earth (download automático)
│
├── intermediate/           # Arquivos intermediários de cache (gerados automaticamente)
│   ├── cadastro.pkl
│   ├── movements2.pkl
│   ├── LST_raster/         # GeoTIFFs mensais de temperatura MODIS
│   ├── road_pressure/      # Saídas shapefile da etapa de roteamento
│   ├── tmp_monthly_flows/  # Agregações mensais de movimentação
│   ├── tmp_road_pressure/  # Arquivos GPKG mensais de pressão em estradas
│   └── r_plots/            # CSVs intermediários de avaliação do modelo
│
├── results/                # Saídas finais do pipeline
│   └── fmd_risk_scored.csv         ← principal entrega
│
├── credentials/            # Segredos OAuth / API (NÃO versionados no git)
│   └── client_secret_*.json
│
├── src/
│   └── paths.py            # Definições centralizadas de caminhos
│
├── run_pipeline.py         # Ponto de entrada principal
└── README.md
```

---

## Instalação

### 1. Clone o repositório

```bash
git clone <repo-url>
cd FMD_surveillace_codesource
```

### 2. Crie e ative um ambiente Python

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Instale as dependências

```bash
pip install pandas geopandas numpy scikit-learn networkx scipy \
            libpysal geobr matplotlib seaborn rasterio geemap \
            earthengine-api pydrive tqdm openpyxl
```

> **Nota:** o `geobr` busca limites municipais do IBGE no primeiro uso; é necessária conexão com a internet.

---

## Configuração de credenciais (somente etapas 2b e 3a)

As etapas 2b (`2_get_temperature_rs_with_raster.py`) e 3a (`3_1_road_animal_preassure.py`)  
exigem acesso ao **Google Earth Engine** e ao **Google Drive**.

1. Autentique no Earth Engine (uma vez):
   ```bash
   earthengine authenticate
   ```
2. Baixe o JSON de segredo OAuth2 no  
   [Google Cloud Console](https://console.cloud.google.com/) e coloque em:
   ```
   credentials/client_secret_<your-id>.json
   ```
   O pipeline procura automaticamente por `credentials/client_secret_*.json`.
   **Nenhuma credencial é versionada no git** (a pasta `credentials/` está no `.gitignore`).
3. Defina o id do seu projeto do Earth Engine via variável de ambiente:
   ```bash
   export GEE_PROJECT=seu-projeto-ee
   ```

---

## Execução do pipeline

### Pipeline completo (etapas 1 – 10)

```bash
python run_pipeline.py
```

### Retomar de uma etapa específica

```bash
python run_pipeline.py --from 7   # reexecuta apenas combinação + análise de risco
```

O executor usa estratégia de **falha rápida**: se qualquer etapa terminar com código diferente de zero,  
o pipeline para imediatamente e mostra o número da etapa que falhou.

---

## Etapas do pipeline

| # | Script | Descrição | Pesado? |
|---|--------|-----------|---------|
| 1 | `1_load_data.py` | Carrega CSVs de cadastro e movimentações GTAS | Não |
| 2 | `2_1_javalis_spatial_distribution.py` | Suavização do risco de javali (pesos espaciais Queen) | Não |
| 3 | `2_get_temperature_rs_with_raster.py` | Download de LST MODIS via Earth Engine + probabilidade de sobrevivência da FA | Sim (download) |
| 4 | `3_1_road_animal_preassure.py` | Roteamento de estradas OSM + exposição de fazendas por buffer | **Muito pesado** (com cache) |
| 5 | `3_get_spatial_net_features.py` | Análise de rede, densidade de fazendas, pressão animal, distâncias a estradas | Moderado |
| 6 | `4_load_and_check_sisbravet_data.py` | Limpeza de dados de vigilância SISBRAVET | Não |
| 7 | `5_distance_to_border.py` | Distância à fronteira internacional (download automático do Natural Earth) | Não |
| 8 | `6_final_score_cadastro_with_sisbravet_labels.py` | Pontuação de biossegurança com ML (dados Epicollect) | Moderado |
| 9 | `7_combine_and_check_data.py` | Consolida todos os conjuntos de atributos | Não |
| 10 | `8_fmd_risk_analysis.py` | Calcula escores compostos de risco + análise de sensibilidade + gráficos | Não |

---

## Saídas esperadas

Após uma execução bem-sucedida, `results/` conterá:

| Arquivo | Descrição |
|---------|-----------|
| `fmd_risk_scored.csv` | **Saída principal** — todas as propriedades com ambos os escores de risco e 9 componentes normalizados |

---

## Comportamento de cache (etapa 4)

A etapa 4 (pressão animal em estradas) constrói um grafo da malha viária e calcula caminhos mínimos  
para cada par de propriedades — isso pode levar várias horas na primeira execução.  
Os resultados intermediários ficam em cache em `intermediate/`:

- `graph_rs_roads.pkl` — grafo OSM bruto
- `graph_rs_roads_simplified.pkl` — grafo contraído (grau 2)
- `routes_roads_cache.pkl` — caminhos mínimos calculados
- `tmp_monthly_flows/*.pkl` — agregações mensais de movimentação
- `tmp_road_pressure/*.gpkg` — camadas mensais de pressão em estradas

Nas execuções seguintes, os arquivos de cache são reutilizados automaticamente.  
Para forçar recomputação completa, exclua os arquivos de cache relevantes em `intermediate/`.

---

## Sistema de caminhos

Todos os caminhos são definidos em [`src/paths.py`](src/paths.py).  
Nenhum caminho absoluto aparece nos scripts do pipeline.  
Todo caminho deriva de `BASE_DIR = Path(__file__).resolve().parent.parent`  
(ou seja, a raiz do projeto), então o pipeline funciona em qualquer máquina sem edições.

---

## Observações

- O pipeline foi desenvolvido para o RS, Brasil, usando formatos de dados governamentais brasileiros.
- A lógica científica (algoritmos, pesos, arquitetura do modelo) permanece igual à versão original de pesquisa.
- As etapas 2b e 4 exigem internet na primeira execução (download via Earth Engine; o auto-download da fronteira do Natural Earth na etapa 7 também precisa de internet).
