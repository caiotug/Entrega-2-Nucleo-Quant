# Entrega-2-Nucleo-Quant

Ciências Econômicas
Caio Tujera Martins
Guilherme Antônio dos Santos Pereira


================================================================================
Relatório de Risco de Portfólio
Núcleo Quant · Liga de Mercado Financeiro · UFU · 2026.1 — Entrega 02
================================================================================

Sobre o Projeto
---------------
Este script realiza uma análise quantitativa completa de risco para um portfólio
de ações brasileiras, cobrindo o período de 2019 a 2024 (incluindo a crise da
Covid-19 e o ciclo de alta de juros).

Portfólio analisado:

  Ticker      Ativo              Setor
  PETR4.SA    Petrobras          Energia
  VALE3.SA    Vale               Mineração
  ITUB4.SA    Itaú Unibanco      Financeiro
  WEGE3.SA    WEG S.A.           Industrial
  BOVA11.SA   ETF Ibovespa       Benchmark

================================================================================
Como Rodar
================================================================================

# 1. Instale as dependências
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
pip install scipy
pip install yfinance

# Ou instale tudo de uma vez:
pip install numpy pandas matplotlib seaborn scipy yfinance

# 2. Execute o script
python relatorio_risco.py

Os gráficos gerados (drawdown.png e fronteira_eficiente.png) serão salvos
na mesma pasta.

================================================================================
Estrutura do Script
================================================================================

  Tarefa 1   VaR 95% e 99% pelos métodos Histórico, Paramétrico e Monte Carlo
  Tarefa 2   CVaR 99% histórico e comparação com o VaR
  Tarefa 3   Drawdown máximo, data do fundo e tempo de recuperação
  Tarefa 4   Otimização de Markowitz e fronteira eficiente (Monte Carlo)

================================================================================
3 Principais Conclusões
================================================================================

1. As caudas do portfólio são pesadas. O CVaR 99% supera significativamente
   o VaR 99%, indicando que, nos piores dias, as perdas vão bem além do limiar
   estatístico. O VaR histórico captura isso melhor do que o método paramétrico
   (Normal), pois incorpora diretamente os retornos extremos observados — como
   o crash de março de 2020.

2. O maior risco do portfólio se materializou na crise de Covid-19. O máximo
   drawdown ocorreu no primeiro trimestre de 2020, com queda superior a 40%,
   refletindo a alta exposição a setores cíclicos (energia, mineração e
   financeiro) e a correlações que aumentam em momentos de crise.

3. A otimização de Markowitz gera ganhos expressivos de eficiência. O portfólio
   de máximo Sharpe supera o equally-weighted em relação risco-retorno,
   comprovando que a alocação ingênua (25% em cada ativo) deixa ganhos de
   diversificação na mesa. Ativos com menor correlação entre si e maior retorno
   ajustado recebem maior peso no portfólio ótimo.

================================================================================
Dependências
================================================================================

  numpy
  pandas
  matplotlib
  seaborn
  scipy
  yfinance
