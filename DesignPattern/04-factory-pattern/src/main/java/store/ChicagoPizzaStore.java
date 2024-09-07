package store;

import ingredient.ChicagoIngredientFactory;
import pojo.*;

public class ChicagoPizzaStore extends PizzaStore {

    @Override
    Pizza createPizza(String type) {
        Pizza pizza = null;
        if (type.equals("cheese")) {
            pizza = new ChicagoCheesePizza(new ChicagoIngredientFactory());
        } else if (type.equals("greek")) {
            pizza = new ChicagoGreekePizza();
        } else if (type.equals("pepperoni")) {
            pizza = new ChicagoPepperoniPizza();
        }
        return pizza;
    }

}
