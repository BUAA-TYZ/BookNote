### 4 工厂模式

##### 简单工厂

```java
public class PizzaStore {
    Pizza orderPizza(String type) {
        Pizza pizza;
        if (type.equals("cheese")) {
            pizza = new CheesePizza();
        } else if (type.equals("greek")) {
            pizza = new GreekePizza();
        } else if (type.equals("pepperoni")) {
            pizza = new PepperoniPizza();
        }
    
        pizza.prepare();
        pizza.bake();
        pizza.cut();
        pizza.box();
        return pizza;
    }
}
```

- 这段代码耦合太严重了，创建 Pizza 的部分和制作的部分在一个函数里
- 将创建 Pizza 的部分提取出来，放入工厂中

```java
public class SimplePizzaFactory {
    public Pizza createPizza(String type) {
        Pizza pizza;
        if (type.equals("cheese")) {
            pizza = new CheesePizza();
        } else if (type.equals("greek")) {
            pizza = new GreekePizza();
        } else if (type.equals("pepperoni")) {
            pizza = new PepperoniPizza();
        }
        return pizza;
    }
}
```

- 随后，我们可以在 `orderPizza()` 中直接调用工厂的方法创建对象
- 我们也可以将 `createPizza()` 声明成静态方法 **(静态工厂)**，这样可以避免将工厂添加到 PizzaStore 类中

##### 工厂方法
- 如果每个工厂制作 Pizza 的方式不一样，那么我们可以将工厂作为一个抽象类，每个具体的工厂负责生产 Pizza

```java
public abstract class PizzaStore {

    abstract Pizza createPizza(String type);

    public Pizza orderPizza(String type) {
        Pizza pizza = createPizza(type);

        pizza.prepare();
        pizza.bake();
        pizza.cut();
        pizza.box();
        return pizza;
    }
}
```

> **工厂方法模式**定义了一个创建对象的接口，但由子类决定实例化哪一个类。工厂方法让类把实例化推迟到子类

> 依赖倒置原则：要依赖抽象，不要依赖具体类
> - 不要让高层组件依赖低层组件，二者都应该依赖抽象
> - PizzaStore 是高层组件，Pizza 实现是低层。一开始的实现，高层依赖具体 Pizza 类

> 抽象工厂方法提供一个接口，用于创建相关或依赖对象的家族，而不需要明确指定具体类
> - 抽象工厂方法的使用是通过对象的组合
> - 而具体的工厂往往会使用工厂方法来完成

- 在代码中，NYCheesePizza **组合** 了 IngredientFactory 用于生产原料，其并不知道会是哪个工厂，这就是 **抽象工厂**
- 而实际的工厂的 `createDough()` 使用了 **工厂方法**