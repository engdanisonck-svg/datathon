# Datathon Passos Mágicos

Projeto de análise dos indicadores educacionais de 2022 a 2024.

## Executar o dashboard

No terminal do VS Code:

```powershell
.venv\Scripts\streamlit.exe run app.py
```

O Streamlit abrirá o endereço local no navegador.

## Atualizar os gráficos estáticos

```powershell
.venv\Scripts\python.exe src\analise_exploratoria.py
```

Os arquivos serão salvos em `outputs/eda/figuras` e `outputs/eda/tabelas`.

## Treinar e avaliar o modelo preditivo

```powershell
.venv\Scripts\python.exe src\modelo_preditivo.py
```

Os resultados, gráficos e o modelo treinado serão salvos em `outputs/modelo`.
