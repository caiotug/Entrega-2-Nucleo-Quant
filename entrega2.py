"""
Relatório de Risco de Portfólio
Núcleo Quant · Liga de Mercado Financeiro · UFU · 2026.1
Entrega 02 – Bloco 2: Risco & Backtesting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURAÇÕES GERAIS
# ─────────────────────────────────────────────
TICKERS      = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "WEGE3.SA", "BOVA11.SA"]
ATIVOS       = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "WEGE3.SA"]   # sem benchmark
START        = "2019-01-01"
END          = "2024-12-31"
POSICAO      = 100_000.0          # R$ 100.000
PESOS_EW     = np.array([0.25, 0.25, 0.25, 0.25])

# ─────────────────────────────────────────────
# DOWNLOAD DE DADOS
# ─────────────────────────────────────────────
print("=" * 60)
print("  RELATÓRIO DE RISCO DE PORTFÓLIO – ENTREGA 02")
print("=" * 60)
print("\nBaixando dados do Yahoo Finance...")

precos_raw = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)["Close"]
precos_raw.dropna(how="all", inplace=True)
precos_raw.ffill(inplace=True)

precos_ativos = precos_raw[ATIVOS].dropna()

# Log-retornos
log_ret_ativos = np.log(precos_ativos / precos_ativos.shift(1)).dropna()

# Retorno diário do portfólio equally-weighted (4 ativos)
retornos_port = log_ret_ativos @ PESOS_EW

print(f"  Período: {log_ret_ativos.index[0].date()} → {log_ret_ativos.index[-1].date()}")
print(f"  Observações: {len(retornos_port)} dias úteis")


# ╔══════════════════════════════════════════════════════════╗
# ║         TAREFA 1: VALUE AT RISK – TRÊS MÉTODOS          ║
# ╚══════════════════════════════════════════════════════════╝
print("\n" + "=" * 60)
print("  TAREFA 1: VALUE AT RISK (VaR) – TRÊS MÉTODOS")
print("=" * 60)

mu    = retornos_port.mean()
sigma = retornos_port.std()

# ── Método 1: Histórico ──────────────────────────────────────
var_hist_99, var_hist_95 = np.percentile(retornos_port, [1, 5])

# ── Método 2: Paramétrico (Normal) ───────────────────────────
var_norm_99, var_norm_95 = stats.norm.ppf([0.01, 0.05], mu, sigma)

# ── Método 3: Monte Carlo ────────────────────────────────────
np.random.seed(42)
simulacoes = np.random.normal(mu, sigma, 50_000)
var_mc_99, var_mc_95 = np.percentile(simulacoes, [1, 5])

# ── Conversão para R$ ────────────────────────────────────────
def to_brl(var_ret):
    return abs(var_ret) * POSICAO

tabela_var = pd.DataFrame({
    "Método"     : ["Histórico", "Paramétrico (Normal)", "Monte Carlo"],
    "VaR 95% (%)" : [f"{var_hist_95*100:.3f}%", f"{var_norm_95*100:.3f}%", f"{var_mc_95*100:.3f}%"],
    "VaR 99% (%)" : [f"{var_hist_99*100:.3f}%", f"{var_norm_99*100:.3f}%", f"{var_mc_99*100:.3f}%"],
    "VaR 95% (R$)": [f"R$ {to_brl(var_hist_95):,.2f}", f"R$ {to_brl(var_norm_95):,.2f}", f"R$ {to_brl(var_mc_95):,.2f}"],
    "VaR 99% (R$)": [f"R$ {to_brl(var_hist_99):,.2f}", f"R$ {to_brl(var_norm_99):,.2f}", f"R$ {to_brl(var_mc_99):,.2f}"],
})

print("\n" + tabela_var.to_string(index=False))

# ── Interpretação ─────────────────────────────────────────────
vars_99 = {"Histórico": var_hist_99, "Paramétrico": var_norm_99, "Monte Carlo": var_mc_99}
mais_conservador = min(vars_99, key=lambda k: vars_99[k])   # mais negativo = maior perda

print(f"""
[INTERPRETAÇÃO – TAREFA 1]
O método mais conservador (maior perda estimada) no VaR 99% é: {mais_conservador}.
  → VaR Histórico  99%: R$ {to_brl(var_hist_99):,.2f}
  → VaR Paramétrico99%: R$ {to_brl(var_norm_99):,.2f}
  → VaR Monte Carlo99%: R$ {to_brl(var_mc_99):,.2f}

Por que o VaR histórico difere do paramétrico com Normal?
  O método paramétrico assume que os retornos seguem uma distribuição Normal,
  que é simétrica e com caudas finas (curtose = 3). Na prática, retornos de
  ativos financeiros apresentam excesso de curtose (caudas pesadas) e assimetria
  negativa – ou seja, eventos extremos de perda ocorrem com frequência maior do
  que a Normal prevê. O VaR histórico captura diretamente esses eventos extremos
  presentes nos dados (ex.: crash da Covid-19 em março/2020), resultando em
  estimativas de risco diferentes, especialmente nas caudas (99%).
""")


# ╔══════════════════════════════════════════════════════════╗
# ║         TAREFA 2: CVaR – O QUE ACONTECE ALÉM DO VaR?   ║
# ╚══════════════════════════════════════════════════════════╝
print("=" * 60)
print("  TAREFA 2: CVaR – O QUE ACONTECE ALÉM DO VaR?")
print("=" * 60)

# ── Cálculo do CVaR 99% histórico ────────────────────────────
var99 = var_hist_99
cvar  = retornos_port[retornos_port <= var99].mean()

# ── Conversão para R$ ────────────────────────────────────────
var99_brl  = abs(var99)  * POSICAO
cvar_brl   = abs(cvar)   * POSICAO
excesso_brl = cvar_brl - var99_brl

print(f"\nVaR 99%:  R$ {var99_brl:,.2f}  |  CVaR 99%:  R$ {cvar_brl:,.2f}  |  Excesso:  R$ {excesso_brl:,.2f}")

razao = cvar_brl / var99_brl

print(f"""
[INTERPRETAÇÃO – TAREFA 2]
O CVaR 99% é {razao:.2f}x maior do que o VaR 99%
  (Excesso de R$ {excesso_brl:,.2f} além do VaR).

Isso indica que a cauda esquerda do portfólio é pesada (fat tail):
  nos dias em que o VaR é violado, as perdas não param no limiar –
  elas vão consideravelmente além, em média {abs(cvar)*100:.2f}% ao dia.

Por que reguladores preferem o CVaR ao VaR como métrica de capital?
  1. O VaR é um limiar binário: diz apenas "há X% de chance de perder
     mais do que R$ Y", mas nada sobre o tamanho da perda acima desse limiar.
  2. O CVaR (Expected Shortfall) mede a perda MÉDIA nos piores cenários,
     sendo uma métrica coerente de risco (satisfaz subaditividade).
  3. Reguladores (Basileia III/IV, FRTB) migraram para o CVaR porque ele
     incentiva melhor a diversificação e captura o risco de cauda – crucial
     para a estabilidade do sistema financeiro em crises como 2008 e Covid-19.
""")


# ╔══════════════════════════════════════════════════════════╗
# ║        TAREFA 3: DRAWDOWN – O PIOR PERÍODO DA HISTÓRIA  ║
# ╚══════════════════════════════════════════════════════════╝
print("=" * 60)
print("  TAREFA 3: DRAWDOWN – O PIOR PERÍODO DA HISTÓRIA")
print("=" * 60)

# ── Série de valor acumulado ──────────────────────────────────
valor_port = (1 + retornos_port).cumprod()

# ── Drawdown ──────────────────────────────────────────────────
pico     = valor_port.cummax()
drawdown = (valor_port - pico) / pico

# ── Máximo Drawdown ───────────────────────────────────────────
max_dd       = drawdown.min()
data_max_dd  = drawdown.idxmin()

# ── Data do pico anterior ao MDD ─────────────────────────────
pico_anterior = valor_port[:data_max_dd].idxmax()

# ── Recuperação: primeiro dia em que volta ao pico ───────────
val_pico = valor_port[pico_anterior]
serie_pos = valor_port[data_max_dd:]
recuperado = serie_pos[serie_pos >= val_pico]

if len(recuperado) > 0:
    data_recuperacao = recuperado.index[0]
    dias_recuperacao = (data_recuperacao - data_max_dd).days
    recuperou_txt = f"Sim – recuperou em {data_recuperacao.date()} ({dias_recuperacao} dias após o fundo)"
else:
    data_recuperacao = None
    dias_recuperacao = None
    recuperou_txt = "Não – o portfólio NÃO se recuperou completamente até o final do período analisado."

print(f"\n  Máximo Drawdown : {max_dd*100:.2f}%")
print(f"  Pico anterior   : {pico_anterior.date()}")
print(f"  Data do fundo   : {data_max_dd.date()}")
print(f"  Recuperação     : {recuperou_txt}")

covid_inicio = pd.Timestamp("2020-02-20")
covid_fundo  = pd.Timestamp("2020-03-23")
coincide = (covid_inicio <= data_max_dd <= pd.Timestamp("2020-05-01"))

print(f"""
[INTERPRETAÇÃO – TAREFA 3]
O maior drawdown coincide com a crise de Covid-19 (fev–mar/2020)? {'SIM' if coincide else 'NÃO'}.
  O fundo ocorreu em {data_max_dd.date()}, muito próximo do mínimo histórico do
  Ibovespa durante a pandemia (~23 mar 2020). A queda reflete o colapso abrupto
  dos mercados globais com o início do lockdown mundial.

Recuperação: {recuperou_txt}
""")

# ── Gráfico de Drawdown ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
ax.fill_between(drawdown.index, drawdown * 100, 0, color="#d62728", alpha=0.55, label="Drawdown")
ax.plot(drawdown.index, drawdown * 100, color="#d62728", linewidth=0.8)
ax.axhline(0, color="black", linewidth=1.0, linestyle="--")
ax.axvline(data_max_dd, color="#1f77b4", linewidth=1.5, linestyle=":", label=f"Máx. Drawdown ({data_max_dd.date()})")
ax.set_title("Drawdown do Portfólio Equally-Weighted (2019–2024)", fontsize=14, fontweight="bold")
ax.set_xlabel("Data")
ax.set_ylabel("Drawdown (%)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("drawdown.png", dpi=150)
plt.show()
print("  Gráfico salvo como 'drawdown.png'")


# ╔══════════════════════════════════════════════════════════╗
# ║     TAREFA 4: OTIMIZAÇÃO DE MARKOWITZ – FRONTEIRA       ║
# ╚══════════════════════════════════════════════════════════╝
print("\n" + "=" * 60)
print("  TAREFA 4: MATRIZ DE COVARIÂNCIA – MAPA DE CALOR")
print("=" * 60)


cov_anual = log_ret_ativos.cov() * 252  

# ── Nomes limpos para exibição ────────────────────────────────
nomes_display = [t.replace(".SA", "") for t in ATIVOS]
cov_display   = cov_anual.copy()
cov_display.index   = nomes_display
cov_display.columns = nomes_display

# ── Matriz de correlação (complementar, para interpretação) ───
corr_matrix = log_ret_ativos.corr()
corr_display = corr_matrix.copy()
corr_display.index   = nomes_display
corr_display.columns = nomes_display

# ── Impressão tabular ─────────────────────────────────────────
print("\n  Matriz de Covariância Anualizada:")
print(cov_display.applymap(lambda x: f"{x:.6f}").to_string())

print("\n  Matriz de Correlação:")
print(corr_display.applymap(lambda x: f"{x:.4f}").to_string())

# ── Estatísticas da covariância por par ───────────────────────
print("\n  Principais pares (correlação):")
pares = []
for i in range(len(nomes_display)):
    for j in range(i + 1, len(nomes_display)):
        corr_val = corr_display.iloc[i, j]
        cov_val  = cov_display.iloc[i, j]
        pares.append((nomes_display[i], nomes_display[j], corr_val, cov_val))

pares.sort(key=lambda x: abs(x[2]), reverse=True)
for a1, a2, corr_v, cov_v in pares:
    sinal = "positiva" if corr_v > 0 else "negativa"
    print(f"    {a1} × {a2}: corr = {corr_v:.4f} ({sinal}) | cov = {cov_v:.6f}")

# ── Par de menor correlação (melhor diversificador) ───────────
par_min = min(pares, key=lambda x: x[2])
par_max = max(pares, key=lambda x: x[2])

# ── Gráfico 1: Heatmap de Covariância ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.set_theme(style="white")

# Painel esquerdo – Covariância
mask_cov = np.zeros_like(cov_display.values, dtype=bool)   # sem máscara (exibe tudo)
hm1 = sns.heatmap(
    cov_display,
    ax=axes[0],
    annot=True,
    fmt=".5f",
    cmap="RdYlGn",
    linewidths=0.8,
    linecolor="white",
    cbar_kws={"label": "Nível de Covariância", "shrink": 0.85},
    annot_kws={"size": 10},
)
axes[0].set_title(
    f"Matriz de Covariância Anualizada\n({START[:4]}–{END[:4]})",
    fontsize=13, fontweight="bold", pad=15
)
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right", fontsize=11)
axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0, fontsize=11)

# Painel direito – Correlação
hm2 = sns.heatmap(
    corr_display,
    ax=axes[1],
    annot=True,
    fmt=".4f",
    cmap="coolwarm",
    vmin=-1, vmax=1,
    center=0,
    linewidths=0.8,
    linecolor="white",
    cbar_kws={"label": "Coeficiente de Correlação", "shrink": 0.85},
    annot_kws={"size": 10},
)
axes[1].set_title(
    f"Matriz de Correlação\n({START[:4]}–{END[:4]})",
    fontsize=13, fontweight="bold", pad=15
)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right", fontsize=11)
axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0, fontsize=11)

plt.suptitle(
    "PETR4 · VALE3 · ITUB4 · WEGE3 – Estrutura de Covariância e Correlação",
    fontsize=14, fontweight="bold", y=1.02
)
plt.tight_layout()
plt.savefig("heatmap_covariancia.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Gráfico salvo como 'heatmap_covariancia.png'")

# ── Gráfico 2: Covariância rolante (janela de 252 dias) ───────
print("\n  Calculando covariâncias rolantes (janela = 252 dias úteis)...")

janela = 252
pares_plot = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
labels_plot = [
    f"{nomes_display[i]}×{nomes_display[j]}"
    for i, j in pares_plot
]

fig, ax = plt.subplots(figsize=(13, 6))
cores = plt.cm.tab10(np.linspace(0, 0.7, len(pares_plot)))

for (i, j), label, cor in zip(pares_plot, labels_plot, cores):
    cov_rolante = (
        log_ret_ativos.iloc[:, i]
        .rolling(janela)
        .cov(log_ret_ativos.iloc[:, j]) * 252
    )
    ax.plot(cov_rolante.index, cov_rolante, label=label, linewidth=1.2, color=cor)

ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax.axvspan(
    pd.Timestamp("2020-02-01"), pd.Timestamp("2020-06-01"),
    color="red", alpha=0.08, label="Covid-19 (fev–jun/2020)"
)
ax.set_title(
    "Covariância Rolante entre Pares de Ativos (janela = 252 dias úteis)",
    fontsize=13, fontweight="bold"
)
ax.set_xlabel("Data")
ax.set_ylabel("Covariância Anualizada")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig("heatmap_covariancia_rolante.png", dpi=150)
plt.show()
print("  Gráfico salvo como 'heatmap_covariancia_rolante.png'")

# ── Interpretação ─────────────────────────────────────────────
print(f"""
[INTERPRETAÇÃO – TAREFA 4]

1. Estrutura de covariância e correlação do portfólio ({START[:4]}–{END[:4]}):
   · Par de MAIOR correlação : {par_max[0]} × {par_max[1]} (corr = {par_max[2]:.4f})
   · Par de MENOR correlação : {par_min[0]} × {par_min[1]} (corr = {par_min[2]:.4f})

2. O que a covariância anualizada indica?
   Covariância mede como dois ativos se movem juntos ao longo do tempo.
   Valores positivos altos indicam que ambos tendem a subir e cair juntos,
   reduzindo o benefício de diversificação. Valores próximos a zero ou negativos
   indicam que os ativos se comportam de forma independente (ou oposta),
   sendo altamente desejáveis numa carteira bem diversificada.

3. Por que a correlação importa mais que a covariância na prática?
   A correlação normaliza a covariância entre −1 e +1, facilitando a comparação
   entre pares independentemente da escala de volatilidade de cada ativo.
   Um par com correlação < 0.3 é considerado um bom diversificador.

4. Dinâmica das covariâncias durante a crise Covid-19 (fev–jun/2020):
   Em períodos de estresse de mercado, as correlações entre ativos tipicamente
   se elevam (fenômeno conhecido como "correlation breakdown" ou "contagion"):
   todos os ativos caem juntos. Isso é visível no gráfico de covariância rolante,
   onde os pares tendem a apresentar picos de covariância em março/2020 –
   justamente quando a diversificação seria mais necessária, ela é menos eficaz.
   Isso reforça a importância de métricas de risco de cauda (CVaR) que capturam
   esses regimes de alta correlação não previstos pela teoria clássica de Markowitz.

5. Implicação para a alocação ótima:
   O par {par_min[0]} × {par_min[1]} (menor correlação = {par_min[2]:.4f}) oferece
   o maior benefício de diversificação. Um gestor pode explorar essa estrutura
   para reduzir a volatilidade do portfólio sem sacrificar retorno esperado.
""")


print("\n" + "=" * 60)
print("  TAREFA 4: OTIMIZAÇÃO DE MARKOWITZ – FRONTEIRA EFICIENTE")
print("=" * 60)

# ── Parâmetros anualizados (252 dias úteis) ───────────────────
ret_anuais = log_ret_ativos.mean() * 252
cov_anual  = log_ret_ativos.cov()  * 252

print("\nRetornos médios anualizados:")
for ticker, r in ret_anuais.items():
    print(f"  {ticker:10s}: {r*100:.2f}%")

# ── Portfólio equally-weighted (referência) ───────────────────
ret_ew  = PESOS_EW @ ret_anuais.values
vol_ew  = np.sqrt(PESOS_EW @ cov_anual.values @ PESOS_EW)
sharpe_ew = ret_ew / vol_ew

# ── Simulação de Monte Carlo: 10.000 portfólios ───────────────
N_SIM = 10_000
np.random.seed(0)

pesos_sim   = np.array([np.random.dirichlet(np.ones(4)) for _ in range(N_SIM)])
rets_sim    = pesos_sim @ ret_anuais.values
vols_sim    = np.sqrt(np.einsum("ij,jk,ik->i", pesos_sim, cov_anual.values, pesos_sim))
sharpes_sim = rets_sim / vols_sim

# ── Portfólio de máximo Sharpe ────────────────────────────────
idx_max     = np.argmax(sharpes_sim)
pesos_max   = pesos_sim[idx_max]
ret_max     = rets_sim[idx_max]
vol_max     = vols_sim[idx_max]
sharpe_max  = sharpes_sim[idx_max]

# ── Gráfico da fronteira eficiente ───────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))

sc = ax.scatter(vols_sim * 100, rets_sim * 100,
                c=sharpes_sim, cmap="viridis", alpha=0.4, s=8, label="Portfólios simulados")
plt.colorbar(sc, ax=ax, label="Índice de Sharpe")

ax.scatter(vol_ew  * 100, ret_ew  * 100, color="red",    s=180, zorder=5,
           edgecolors="black", linewidth=1.2, label="Equally-Weighted (25% cada)")
ax.scatter(vol_max * 100, ret_max * 100, color="gold",   s=220, zorder=5,
           edgecolors="black", linewidth=1.2, label=f"Máx. Sharpe ({sharpe_max:.2f})")

ax.set_title("Fronteira Eficiente de Markowitz – Monte Carlo (10.000 portfólios)", fontsize=13, fontweight="bold")
ax.set_xlabel("Volatilidade Anualizada (%)")
ax.set_ylabel("Retorno Anualizado (%)")
ax.legend(loc="upper left")
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("fronteira_eficiente.png", dpi=150)
plt.show()
print("  Gráfico salvo como 'fronteira_eficiente.png'")

# ── Impressão dos pesos ───────────────────────────────────────
print("\n  Pesos do Portfólio de Máximo Sharpe vs. Equally-Weighted:")
print(f"  {'Ativo':<12} {'EW':>8} {'Máx. Sharpe':>14}")
print("  " + "-" * 36)
for ticker, w_ew, w_ms in zip(ATIVOS, PESOS_EW, pesos_max):
    print(f"  {ticker:<12} {w_ew*100:>7.1f}%  {w_ms*100:>12.1f}%")

print(f"""
  Métricas Comparativas:
  {'Portfólio':<22} {'Retorno':>10} {'Volatilidade':>14} {'Sharpe':>10}
  {'Equally-Weighted':<22} {ret_ew*100:>9.2f}%  {vol_ew*100:>12.2f}%  {sharpe_ew:>9.2f}
  {'Máximo Sharpe':<22} {ret_max*100:>9.2f}%  {vol_max*100:>12.2f}%  {sharpe_max:>9.2f}
""")

ativo_maior_peso = ATIVOS[np.argmax(pesos_max)]

print(f"""[INTERPRETAÇÃO – TAREFA 4]
O portfólio de máximo Sharpe apresenta Sharpe = {sharpe_max:.2f}, contra {sharpe_ew:.2f}
do equally-weighted – uma melhora de {(sharpe_max/sharpe_ew - 1)*100:.1f}% na relação risco-retorno.

O ativo com maior alocação no portfólio ótimo é {ativo_maior_peso} ({pesos_max[np.argmax(pesos_max)]*100:.1f}%).
  Isso ocorre pois a otimização de Markowitz privilegia ativos que oferecem
  maior retorno por unidade de risco E que possuem baixa correlação com os demais,
  permitindo reduzir a volatilidade total do portfólio via diversificação.

A diferença entre o equally-weighted e o ótimo evidencia que alocação
ingênua (pesos iguais) deixa retorno ajustado ao risco na mesa – a teoria
de Markowitz captura ganhos de diversificação que o EW ignora.
""")
# ╔══════════════════════════════════════════════════════════╗
# ║               RESUMO EXECUTIVO FINAL                    ║
# ╚══════════════════════════════════════════════════════════╝
print("=" * 60)
print("  RESUMO EXECUTIVO – RELATÓRIO DE RISCO")
print("=" * 60)
print(f"""
PORTFÓLIO: PETR4 | VALE3 | ITUB4 | WEGE3 (equally-weighted, R$ 100.000)
PERÍODO  : 2019-01-01 a 2024-12-31

1. VaR & CVaR
   · VaR 99% Histórico   : R$ {to_brl(var_hist_99):,.2f} por dia
   · CVaR 99% Histórico  : R$ {cvar_brl:,.2f} por dia
   · O CVaR supera o VaR em R$ {excesso_brl:,.2f}, sinalizando caudas pesadas.

2. Drawdown
   · Máximo Drawdown     : {max_dd*100:.2f}%  (fundo em {data_max_dd.date()})
   · Coincidiu com o crash da Covid-19 (fev–mar 2020).
   · {recuperou_txt}

3. Markowitz / Fronteira Eficiente
   · Sharpe EW           : {sharpe_ew:.2f}
   · Sharpe Ótimo        : {sharpe_max:.2f}  (+{(sharpe_max/sharpe_ew - 1)*100:.1f}% de melhora)
   · Ativo dominante no ótimo: {ativo_maior_peso} ({pesos_max[np.argmax(pesos_max)]*100:.1f}%)

4. Matriz de Covariância / Correlação
   · Par mais correlacionado : {par_max[0]} × {par_max[1]} (corr = {par_max[2]:.4f})
   · Melhor diversificador   : {par_min[0]} × {par_min[1]} (corr = {par_min[2]:.4f})
   · Covariâncias sobem em crises (Covid-19), reduzindo benefício da diversificação.
   · Gráficos gerados: heatmap_covariancia.png | heatmap_covariancia_rolante.png
""")
print("=" * 60)
print("  Script executado com sucesso. Gráficos salvos.")
print("=" * 60)
