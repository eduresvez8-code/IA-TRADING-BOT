"""Ajuste de cantidades y precios a los filtros de microestructura de Binance.

Binance rechaza una orden cuya cantidad no sea múltiplo de stepSize (LOT_SIZE) o
cuyo precio no sea múltiplo de tickSize (PRICE_FILTER). Aquí vive ESE ajuste, y
se hace con `decimal.Decimal` —nunca con float—: en binario `0.1 + 0.2 != 0.3`,
y un floor mal hecho violaría el riesgo o el saldo libre por unos satoshis.

Ejemplo del bug que esto evita: en float, `0.3 // 0.1 == 2.0` (porque
`0.3/0.1 == 2.9999…`); con Decimal da 3, que es lo correcto.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP


def floor_to_step(qty: float | Decimal, step: Decimal) -> Decimal:
    """Trunca `qty` al múltiplo de `step` inmediatamente inferior (jamás arriba).

    Redondear hacia arriba aumentaría la cantidad y, con ella, el riesgo y el
    capital comprometido — exactamente lo que el Risk Manager debe impedir.
    """
    q = qty if isinstance(qty, Decimal) else Decimal(str(qty))
    # `//` en Decimal trunca hacia cero; con operandos positivos equivale a floor.
    return (q // step) * step


def round_to_tick(price: float | Decimal, tick: Decimal) -> Decimal:
    """Redondea un precio al múltiplo de `tick` más cercano (half-up).

    Para SL/TP el sentido del redondeo es indiferente: el Risk Manager recalcula
    la distancia real al stop DESPUÉS de redondear, así el tamaño de la posición
    siempre corresponde al stop que de verdad se va a colocar.
    """
    p = price if isinstance(price, Decimal) else Decimal(str(price))
    return (p / tick).quantize(Decimal(1), rounding=ROUND_HALF_UP) * tick
