package pojo;

import ingredient.PizzaIngredientFactory;

public class ChicagoCheesePizza extends Pizza {

    private PizzaIngredientFactory ingredientFactory;

    public ChicagoCheesePizza(PizzaIngredientFactory ingredientFactory) {
        this.ingredientFactory = ingredientFactory;
        name = "Chicago Cheese Pizza";
        dough = "Thick Crust Dough";
        sauce = "Plum Tomato Sauce";

        toppings.add("Shredded Mozzarella Cheese");
    }

    @Override
    public void prepare() {
        ingredientFactory.createDough();
    }

    public void cut() {
        System.out.println("Cutting the pizza into square slices");
    }
}
