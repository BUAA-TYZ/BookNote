package ingredient;

import pojo.Pizza;

public class NYIngredientFactory implements PizzaIngredientFactory {

    @Override
    public Dough createDough() {
        return new ThinDough();
    }
}
