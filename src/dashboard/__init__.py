"""Dashboard de observabilidad en tiempo real (proceso READ-ONLY).

Lee la misma SQLite que escribe el engine, en modo `ro` (físicamente incapaz de
escribir), y la sirve como JSON a un frontend de una sola página. NUNCA toca el
exchange ni envía órdenes: la regla "toda orden pasa por risk/manager → execution"
se respeta de forma trivial porque el dashboard no conoce al executor.
"""
