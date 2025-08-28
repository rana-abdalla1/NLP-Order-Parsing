from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Topping:
    NOT: bool
    Quantity: Optional[str] = None
    Topping: Optional[str] = None

@dataclass
class Style:
    NOT: bool
    TYPE: Optional[str] = None

@dataclass
class OnePizzaOrder:
    NUMBER: Optional[str] = None
    SIZE: Optional[str] = None
    STYLE: List[Style] = field(default_factory=list)
    AllTopping: List[Topping] = field(default_factory=list)

@dataclass
class PizzaOrder:
    orders: List[OnePizzaOrder] = field(default_factory=list)

@dataclass
class OneDrinkOrder:
    NUMBER: Optional[str] = None
    SIZE: Optional[str] = None
    DRINKTYPE: Optional[str] = None
    CONTAINERTYPE: Optional[str] = None

@dataclass
class DrinkOrder:
    orders: List[OneDrinkOrder] = field(default_factory=list)

#! Placeholders for the data postprocessing
class DataPostprocessing:
    def __init__(self):
        self.pizza_order = PizzaOrder()
        self.drink_order = DrinkOrder()

    def add_pizza_order(self, order):
        self.pizza_order.orders.append(order)

    def add_drink_order(self, order):
        self.drink_order.orders.append(order)

    def get_pizza_order(self):
        return self.pizza_order

    def get_drink_order(self):
        return self.drink_order