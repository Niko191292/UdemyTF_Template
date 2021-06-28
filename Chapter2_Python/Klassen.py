class Car:
    # Konstruktor self = this
    def __init__(self, name: str, jahr: int, ps: int) -> None:
        self.name = name
        self.jahr = jahr
        self.ps = ps

    def print_car(self) -> None:
        print("Name vom Auto: ", self.name)
        print("PS vom Auto: ", self.ps)


my_car = Car("Porsche 911", 2019, 400)
my_car.print_car()

