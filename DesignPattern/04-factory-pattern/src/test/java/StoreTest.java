import org.junit.jupiter.api.Test;
import store.ChicagoPizzaStore;
import store.NYPizzaStore;
import store.PizzaStore;

public class StoreTest {

    @Test
    public void testStore() {
        PizzaStore pizzaStore1 = new NYPizzaStore();
        pizzaStore1.orderPizza("cheese");
        PizzaStore pizzaStore2 = new ChicagoPizzaStore();
        pizzaStore2.orderPizza("cheese");
    }
}
