from ex01.src.pattern import Spectrum
from ex01.src.pattern import Checker
from ex01.src.pattern import Circle
from ex01.src.generator import ImageGenerator

# do stuff!

checker = Checker(12, 2)
checker.draw()
checker.show()

circle = Circle(12, 3, [3,4])
circle.draw()
circle.show()

spec = Spectrum(32)
spec.draw()
spec.show()

gen = ImageGenerator('./data/exercise_data/','./data/Labels.json',60, [32, 32, 3])
gen.show()

