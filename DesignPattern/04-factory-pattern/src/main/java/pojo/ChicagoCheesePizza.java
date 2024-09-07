package pojo;

public class ChicagoCheesePizza extends Pizza {

    public ChicagoCheesePizza() {
        name = "Chicago Cheese Pizza";
        dough = "Thick Crust Dough";
        sauce = "Plum Tomato Sauce";

        toppings.add("Shredded Mozzarella Cheese");
    }

    public void cut() {
        System.out.println("Cutting the pizza into square slices");
    }
}
