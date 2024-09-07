package store;

import ingredient.NYIngredientFactory;
import pojo.NYCheesePizza;
import pojo.NYGreekePizza;
import pojo.NYPepperoniPizza;
import pojo.Pizza;

public class NYPizzaStore extends PizzaStore {

    @Override
    Pizza createPizza(String type) {
        Pizza pizza = null;
        if (type.equals("cheese")) {
            pizza = new NYCheesePizza(new NYIngredientFactory());
        } else if (type.equals("greek")) {
            pizza = new NYGreekePizza();
        } else if (type.equals("pepperoni")) {
            pizza = new NYPepperoniPizza();
        }
        return pizza;
    }
}
