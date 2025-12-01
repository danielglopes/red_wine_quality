# Repository Guidelines

## Project Structure & Module Organization
- `src/data_prep.py` preprocessa o CSV em `data/raw/winequality-red.csv`, normaliza atributos e salva `*.npy` prontos em `data/processed/`.  
- `src/models/` reserva scripts de treinamento (ex.: `adaline.py`, `perceptron.py`), úteis para notebooks ou CLIs simples.  
- `src/visualization.py` deve centralizar gráficos gerados a partir de arrays processados.  
- `results/` guarda saídas geradas (figuras, pesos/artefatos de modelos). Evite sobrescrever sem versionar nomes datados.  
- Dados de entrada permanecem em `data/raw/`; não edite manualmente para garantir reprodutibilidade.

## Build, Test, and Development Commands
- Criar ambiente: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.  
- Preparar dados (gera `data/processed/*.npy`): `python src/data_prep.py`.  
- Rodar notebooks/Lab se necessário: `jupyter lab` (usa ambiente ativo).  
- Limpeza rápida (remover processados): `rm -f data/processed/*.npy` antes de refazer o pipeline.

## Coding Style & Naming Conventions
- Python com indentação de 4 espaços; siga PEP8.  
- Funções/métodos em snake_case (`carregar_dados`); classes em PascalCase.  
- Use docstrings curtas (pt-BR já adotado) explicando I/O; adicione comentários apenas para trechos não óbvios.  
- Prefira caminhos relativos via `os.path` como em `data_prep.py` para manter portabilidade.  
- Inclua anotações de tipo quando possível para arrays/shape esperados.

## Testing Guidelines
- Ainda não há suíte de testes; ao adicionar, use `pytest` com arquivos `tests/test_*.py`.  
- Foque em testar: (1) leitura e normalização do pipeline, (2) divisões estratificadas, (3) métricas dos modelos.  
- Gere dados sintéticos pequenos para testes; não dependa de `data/raw` real.  
- Rodar tudo: `pytest`.

## Commit & Pull Request Guidelines
- Mensagens curtas em imperativo (ex.: “Add processed data files”), 72 caracteres no título; detalhe racional em corpo opcional.  
- Inclua descrição clara no PR: objetivo, principais mudanças, como validar (comandos), e se altera dados em `results/`.  
- Anexe figuras/artefatos relevantes produzidos em `results/figures` ou `results/models`; evite anexar binários grandes no repositório.  
- Relacione issues/tarefas no PR e liste riscos conhecidos ou passos faltantes.
