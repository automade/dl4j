import org.scalatest._

class HelloSpec extends FunSuite with DiagrammedAssertions {
  test("Hello starts with H") {
    assert("Hello".startsWith("H"))
  }
}
