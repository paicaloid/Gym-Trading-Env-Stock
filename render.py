from src.stock_trading_env.renderer import Renderer


renderer = Renderer(render_logs_dir="render_logs")

# renderer.add_line( name= "avg_price", function= lambda df : df["avg_price"], line_options ={"width" : 1, "color": "purple"})

renderer.add_metric(
    name="total number of sell",
    function=lambda df : f"{df.loc[df['realized_pnl'] != 0, ['realized_pnl']].count()['realized_pnl']:0.2f}"
)

renderer.add_metric(
    name="win sell",
    function=lambda df : f"{df.loc[df['realized_pnl'] > 0, ['realized_pnl']].count()['realized_pnl']:0.2f}"
)

renderer.add_metric(
    name="loss sell",
    function=lambda df : f"{df.loc[df['realized_pnl'] < 0, ['realized_pnl']].count()['realized_pnl']:0.2f}"
)

renderer.add_metric(
    name="percent profitable",
    function=lambda df : f"{df.loc[df['realized_pnl'] > 0, ['realized_pnl']].count()['realized_pnl'] / df.loc[df['realized_pnl'] != 0, ['realized_pnl']].count()['realized_pnl'] * 100:0.2f}%"
)

renderer.add_metric(
    name="gross profits",
    function=lambda df : f"{df.loc[df['realized_pnl'] > 0, ['realized_pnl']].sum()['realized_pnl']:0.2f}"
)

renderer.add_metric(
    name="gross loss",
    function=lambda df : f"{df.loc[df['realized_pnl'] < 0, ['realized_pnl']].sum()['realized_pnl']:0.2f}"
)
renderer.add_metric(
    name="profit factor",
    function=lambda df : f"{df.loc[df['realized_pnl'] > 0, ['realized_pnl']].sum()['realized_pnl'] / df.loc[df['realized_pnl'] < 0, ['realized_pnl']].sum()['realized_pnl'] * -1:0.2f}"
)

renderer.run()
