### GOAL: Going from an order to a planner

food_opts = ["chicken", "sandwich"]

def get_order():
    order = (input("What would you like to order? ")).lower()
    plan = []

    is_sandwich = False
    for food in food_opts:
        if food in order:
            if food == "sandwich":
                is_sandwich = True
            else:
                plan.append(food)

    plan.insert(0, "bread")
    plan.insert(len(plan), "bread")

    return plan
