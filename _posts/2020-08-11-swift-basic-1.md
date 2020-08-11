---
classes: wide
title: "Swift Basic-1"
date: 2020-08-11 15:00:00 -0400
categories: apple swift
---

# 목차
[0. Swift](#swift)
[1. Basic](#basic)
[2. References](#references)
<br>

# Swift
Swift는 iOS, macOS, watchOS 그리고 tvOS 앱 개발을 위한 새로운 프로그래밍 언어다. Swift의 많은 부분은 C와 Objective-C 개발 경험과 상당히 유사하다. 

Swift는 `Int`, `Double`, `Float`, `Bool`, `String`을 포함한 C와 Objective-C의 기초적인 기능들과 함께  `Array`, `Set` 그리고 `Dictionary`와 같은 강력한 콜렉션 기능을 제공한다.

Swift는 C와 같이 값을 저장하고 참조할 수 있는 변수(Variable)들을 사용하면서 변수(Varibale)의 값이 변경될 수 없도록 하는 확장 기능을 포함한다. 이러한 확장 기능은 상수(Constant)라고 알려져 있고, C보다 강력한 기능이다. 상수(Constant)는 Swift에서 코드를 보다 안전하고 분명하게 하기 위해서 사용되어진다. 

추가적으로 Swift는 C와 Objective-C에서는 찾을 수 없는 `Tuple`과 같은 type을 제공한다. `Tuple`은 개발자가 값을 그룹처럼 만들고 관리할 수 있게 한다. 개발자는 `Tuple`을 통해 함수(Function)에서 하나의 값처럼 여러 개의 값을 반환할 수 있다.

또한 Swift는 값이 없는 것을 다루는 `Optional` Type을 제공한다. `Optional`은 "여기에 있는 값은 x와 같아" 또는 "여기에는 값이 없어"와 같이 알려준다. `Optional`을 사용하는 것은 Objective-C에서 포인터 안의 `nil`을 사용하는 것과 비슷하지만, 클래스 뿐만 아니라 모든 type에서 동작한다. `Optional`는 Objective-C의 `nil` 포인터 보다 안전하고 명확할 뿐만 아니라 Swift의 많은 핵심적인 기능들의 심장부에 있다.

Swift는 개발자가 사용하고 있는 값의 Type이 무엇인지 분명하게 확인할 수 있도록 해주는 **Type-Safe** 언어다. 만약, 개발자의 코드의 하나의 부분이 `String`이 필요하다면, 실수로 개발자가 이것을 `Int`로 처리하는 것을 방지해준다. 마찬가지로 `Non-Optional String`을 필요로 하는 코드의 일부분에 우연히 `optional String`이 처리되는 것을 방지해준다. **Type-Safety**는 개발자가 개발 과정에서 가능한 초기에 오류를 찾고 고치도록 도와준다.
<br>

# Basic
## 상수(Constant) and 변수(Variable)
### Declare
```swift
let maximumNumberOfLoginAttempts = 10
var currentLoginAttempt = 0

var x = 0.0, y = 0.0, z = 0.0
```
### Type Annotation
상수(Constatant) 또는 변수(Variable)를 선언할 때 저장할 수 있는 값의 형태를 분명하게 하기 위해서 **Type Annotation**을 통해 선언할 수도 있다.
```swift
var welcomeMsg: String
welcomeMsg = "Hello"

var red, green, blue: Double
```
### Naming
Swift에서는 유니코드(Unicode) 문자를 포함한 거의 대부분의 문자를 포함할 수 있다. 하지만 공백, 수학 기호, 화살표, 개인적으로 사용하는 유니코드 값, 박스 드로잉 기호 등은 포할할 수 없다. 또한, 숫자로 시작될 수 없다.
```swift
let π = 3.14159
let 你好 = "你好世界"
let 🐶🐮 = "dogcow"
```
### Changing
변수(Variable)와는 달리 상수(Constant)는 설정된 값을 변경할 수 없다. 만약 다음과 같이 시도할 경우, 컴파일 시 에러가 발생할 것이다.
```swift
var friendlyWelcome = "Hello!"
friendlyWelcome = "Bonjour!"

let languageName = "Swift"
languageName = "Swift++" // ERROR
```
### Print
```swift
print(friendlyWelcom)
print("The currnet value of friendlyWelcome is \(friendlyWelcome)")
```
## Comments
```swift
// This is a comment
```
```swift
/* This is also comment
but is written over multiple lines. */
```
```swift
/* This is the start of the first multiline comment.
 /* This is the second, nested multiline comment. */
This is the end of the first multiline comment. */
```
## Semicolons
많은 개발 언어와는 다르게 Swift는 원하더라도 코드의 문장 끝에 Semicolon(;)을 사용할 수 없다. 하지만, 여러 줄의 코드를 하나의 줄에 적을 때는 가능하다.
```swift
let cat = "🐱"; print(cat)
```
## Type Safety and Type Inference
Swift는 **Type Safe** 언어이기 때문에 `String`이 필요한 부분에 실수로 `Int`를 통과시킬 수 없다. 왜냐하면 컴파일을 하면서 코드와 플래그가 오류로서 잘 못 매칭되어져 있는지를 확인하는 과정을 거치기 때문이다. 이러한 기능은 개발자가 개발 과정에서 가능한 빠르게 오류를 해결할 수 있도록 했다. 

Type 검사는 개발자가 다른 Type의 값을 동작시킬 때 오류를 피하도록 도와주지만 매번 선언하는 상수(Constant)나 변수(Variable)의 Type을 개발자가 지정해야한다는 것은 아니다. 개발자가 필요한 값의 Type을 지정하지 않는다면 Swift는 **Type Inference**를 통해 적절한 Type을 사용한다. **Type Inference**는 컴파일을 하면서 자동적으로 특정 표현의 Type을 컴파일러가 추론할 수 있게한다. 
```swift
let meaningOfLife = 42 // Int
let pi = 3.142592 // Double
let anotherPi = 3 + 0.141592 // Double
```
## Numeric Literals
```swift
let decimalInterger = 17
let binaryInteger = 0b10001
let octalInteger = 0o21
let hexadecimalInteger = 0x11

let decimalDouble = 12.1875
let exponentDouble = 1.21857e1
let hexadecimalDouble = 0xc.3p0

let paddedDouble = 000123.456
let oneMillion = 1_000_000
let justOverOneMilion = 1_000_000.000_000_1
```
## Numeric Type Conversion
편의를 위해서 기본 `Int` Type을 사용하는 것도 좋지만 제약된 크기의 데이터 사용 및 최적화를 위해서 필요한 데이터의 Type을 지정해서 사용하는 것도 하나의 방법이다. 정해진 데이터 Type을 사용하는 것은 갑작스러운 Data Overflow를 잡고 사용되는 데이터의 특성을 파악할 수 있다.
### Integer Conversion
```swift
let twoThousand: UInt16 = 2_000
let one: UInt8 = 1
let twoThousandAndOne = twoThousand + UInt16(one)
```
### Integer to Floating-Point Conversion
```swift
let three = 3
let pointOneFourOneFiveNine = 0.14159
let pi = Double(three) + pointOneFourOneFiveNine // 3.14159
let integerPi = Int(pi) // 3
```
## Type Aliases
**Type Aliases**는 존재하는 Type의 이름을 대체할 별칭을 하나 선언하는 것으로 기존의 이름보다 문맥적으로 적절하다.
```swift
typealias AudioSample = UInt16
var maxAmplitudeFound = AudioSample.min
```
## Boolean
```swift
let orangesAreOragen = true
let turnipsAreDelicious = false
let i = 1

if turnipsAreDelicious {
    print("Mmm, tasty turnips!")
} else {
    print("Eww, turnips are horrible.")
}

if i == 1 {
    print("The number of i is \(i)")
}
```
## Tuple
`Tuple`은 하나의 값 안에 여러 개의 값을 그룹짓는 것이다. `Tuple` 안의 값들은 어느 타입이든 될 수 있고 서로 같은 Type일 필요는 없다. 
```swift
let http404error = (404, "Not Found")
let http200Status = (statusCode: 200, description: "OK")
let (statusCode, statusMsg) = http404error

print("The status code is \(statusCode)")
print("The status message is \(statusMsg)")

print("The status code is \(http200Status.statusCode)")
print("The status message is \(http200Status.description)")

print("The status code is \(http404error.0)")
print("The status message is \(https404error.1)")
```
## Optional
값이 없는 상황에서 개발자는 `Optional`을 사용할 수 있다. `Optional`은 "값이 존재하고 접근하기 위해서는 풀어야한다" 또는 "값이 없다" 라는 두 가지의 가능성을 의미한다.

Swift의 `Int` Type은 기본적으로 `String`을 `Int`로 변환할려는 Initializer를 가지고 있다. 하지만 모든 `String`이 `Int`이 될 수 없다. 왜냐하면 문자열 "123"은 숫자 123으로 변환이 가능하지만 "Hello, World"와 같은 문자열은 명확한 숫자로 변환될 수 없기 때문이다. 따라서 `String`을 `Int`로 바꾸면서 `Optional Int`인 `Int?`를 반환한다. 
```swift
let possibleNumber = "123"
let convertedNumber = Int(possibleNumber)
```
### nil
개발자는 특별한 값인 `nil`을 할당함으로써 값이 없는 상태에 `Optional` 변수(Variable)를 선언할 수 있다.
```swift
var serverResponseCode: Int? = 404 // variable contains an actual Int value of 404
serverResponseCode = nil // variable contains no value

var surveyAnswer: String? // variable is automatically set to nil
```
### Forced Unwrapping
만약 `Optional`에 값이 있다고 확신이 된다면, `Optional`의 이름 뒤에 느낌표(\!)를 추가함으로써 실제 값에 접근할 수 있다. 느낌표(\!)는 "나는 Optional에 값이 있다는 것을 확실히 알고 있고, 사용할 것이다"라는 것을 의미한다. 
```swift
let strNum = "123"
let intNum = Int(strNum)

if intNum != nil {
    print("strNum has an integer value of \(intNum!)")
}
```
### Optional Binding
개발자는 `Optional`이 값을 가지고 있는지 아닌지를 파악하거나 일시적인 상수(Constant)나 변수(Variable)로 이용가능한 값을 만들기 위해 **Optional Binding**을 사용할 수 있다. **Optional Binding**은 `Optional` 안에 값을 확인하거나 상수(Constant)나 변수(Variable) 안에 값을 얻기 위해서 `if`나 `while`을 사용할 수도 있다.
```swift
if let constantName = someOptional {
    statements
}
```
```swift
if let actualNumber = Int(possibleNumber) {
    print("The string \(possibleNumber) has an integer value of \(actualNumber)")
} else {
    print("The string \(possibleNumber) could not be converted to an integer")
}
```
개발자는 Commas(,)로 구분하여 하나의 `if`문에 조건(Condition)과 **Optional Binding**을 포함시킬 수 있다.
```swift
if let firstNumber = Int("4"), let secondNumber = Int("42"),
firstNumber < secondNumber && secondNumber < 100 {
    print("\(firstNumber) < \(secondNumber) < 100")
}

if let firstNumber = Int("4"){
    if let secondNumber = Int("42"){
        if firstNumber < secondNumber && secondNumber < 100 {
            print("\(firstNumber) < \(secondNumber) < 100")
        }
    }
}
```
### Implicitly Unwrapped Optionals
위에서 언급했듯이 `Optional`은 값이 없을 수도 있기 때문에 `if`를 통해 값을 확인하고 조건적으로 Optional Binding으로 값에 접근할 수 있다. 하지만 가끔씩 `Optional`이 항상 값을 가지고 있는 구조가 확실하다면 이러한 과정을 수행하지 않는 것이 효율적이다. 

이러한 `Optional`의 특징을 **Implicitly Unwrapped Optionals**이라고 정의되어져 있다. 개발자는 `String?`을 `String!`으로 선언함으로써 사용할 수 있다.
```swfit
let possibleString: String? = "An optional string."
let forcedString: String = possibleString!

let assumedString: String! = "An implicitly unwrapped optional string."
let implicitString: String = assumedString

if assumedString != nil{
    print(assumedString!)
}

if let definiteString = assumedString {
    print(definiteString)
}
```

개발자는 **Implicitly Unwrapped Optionals**이 `Optional`을 강제로 푸는 권한을 주는 것으로 생각할 수 있다.  개발자가 **Implicitly Unwrapped Optionals**를 사용할 때, Swift는 먼저 순차적으로 `Optional` 값으로서 사용한다. 만약 `Optional`로 사용되지 않았다면, Swift는 강제로 값을 푼다. 위 코드에서 볼 수 있듯이, `Optional` 값 `assuemdString`은 `implicitString`에 값으로서 할당되기 전에 강제로 풀려진다. 왜냐하면 `implicitString`은 분명한 Type인 `Non-Optional String`을 가졌기 때문이다. 하지만 아래 코드에서 보듯이, `optionalString`은 분명한 Type을 가지고 있지 않다.
```swift
let optionalString = assumedString 
// The type of optionalString is "String?" and assumedString isn't force-unwrapped.
```
## Error Handling
개발자는 실행 과정에서 직면할 수 있는 오류 조건들에 대한 **Error Handling**을 사용할 수 있다. `Optional`과는 다르게 **Error Handling**은 개발자가 실패의 원인을 기반으로 결정하고 필요하다면, 프로그램의 다른 부분으로 오류를 전달할 수 있도록 한다.

함수(Fucntion)의 선언에서 `throws` 키워드를 포함시킴으로써 이 함수(Function)는 오류를 던질 수 있는 기능을 가진다. Swift는 자동적으로 오류가 `catch` 구문에서 다루어질 때까지 현재 범위의 밖으로 오류를 전달한다. `do`는 새로운 범위를 만들면서 오류가 하나 또는 여러 개의 `catch` 구문에 전달될 수 있도록 한다.
```swift
func canTrhowAnError() throws {
    // this function may or may not throw an error
}

do {
    try canTrhowAnError()
    // no error was thrown
} catch {
    // an error was thrown
}
```
```swift
func makeASandwich() throws{
    // . . .
}

do {
    try makeASandwich()
    eatASandwich()
} catch SandwichError.outOfCleanDishes {
    washDishes()
} catch SandwichError.missingIngredients(let ingredients) {
    buyGroceries(ingredients)
}
```
## Assertions and Preconditions
**Assertions**와 **Preconditions**는 런타임에서 발생하는 것에 상황에 대해서 확인할 수 있는 방법이다. 개발자는 어떤 코드의 실행 전에 필수적인 조건을 확실하게 하기 위해서 이러한 기능들을 사용할 수 있다. 만약 **Assertions** 또는 **Preconditions** 안에서 `Bool` 조건이 `true`라면, 코드는 평소대로 계속 진행될 것이다. 만약 `false`라면, 프로그램의 현재 상태는 유효하지 않기 때문에 프로그램은 중단된다.

개발자는 만들 수 있는 가정과 코딩 중에 가질 수 있는 예상을 표현하기 위해서 **Assertions**와 **Preconditions**를 사용하고, 코드에 이러한 기능들을 포함시킬 수 있다. **Assertions**는 개발자가 실수와 개발 과정에서 정확하지 않은 가정을 발견할 수 있도록 도와주고, **Preconditions**는 Production Issues를 파악할 수 있도록 도와준다.

추가적으로 런타임에서 개발자가 예상한 것을 검증하기 위해서, **Assertions**와 **Preconditions**는 코드 안에서 유용한 도큐먼트의 형식이 된다. 위에서 언급한 Error Handling과는 다르게, **Assertions**와 **Preconditions**는 회복가능하거나 예상된 오류에 사용되지 않는다. 왜냐하면 실패된 **Assertion** 또는 **Precondition**은 유효하지 않은 프로그램 상태를 의미하기 때문이다. 이러한 실패한 **Assertion**을 잡을 수 있는 방법은 없다. 

**Assertions**와 **Preconditions**은 개발자의 코드를 유효하지 않은 조건이 발생하지 않도록 하는 대용품으로서 사용할 수 없다. 하지만 만약 유효하지 않은 상태가 발생한다면 유효한 데이터와 상태 원인을 예측 가능하게 중지시키고 문제를 보다 쉽게 디버그 할 수 있도록 만들어주는데 사용할 수 있다. 유효하지 않은 상태가 발견되자마자 실행을 멈추는 것은 유효하지 않은 상태가 원인이 되는 데미지를 제한할 수 있도록 도와준다.

**Assertions**과 **Precondtions**는 이들이 확인될 때 차이점이 존재한다. **Assertions**은 디버그 빌드에서만 확인되어지지만, **Preconditions**은 디버그와 Production 빌드 둘 다에서 확인될 수 있다. Production 빌드에서 **Assertions** 안의 조건은 평가되지 않는다. 이것은 사용자가 Production 성능에 영향을 주지 않고 개발 과정동안 개발자가 원하는 만큼 많은 **Assertions**을 사용할 수 있다는 것을 의미한다.
### Debugging with Assertions
개발자는 `true`인지 `false`인지 평가하는 표현에 함수(Function)를 통과시키고 만약 결과가 `false`라면 지정된 메세지를 출력된다.
```swift
let age = -3
assert(age>=0, "A person's age can't be less than zero.")

if age > 10{
    print("You can ride the roller-coaster or the ferris wheel.")
} else if age >= 0 {
    print("You can ride the ferris wheel.")
} else {
    assertionFailure("A person's age can't be less than zero.")
}
```
### Enforcing Preconditions
반드시 실행을 계속 진행하기 위해서 코드의 조건이 반드시 `true`가 되어야하지만, 언제든지 잠재적으로 `false`가 될 수 있는 조건을 가지는 경우 Precondition을 사용한다. 예를 들면, 인덱스의 범위를 벗어나지 않는 경우 또는 함수(Function)에 유효한 값이 전달되어야 하는 경우가 있다.
```swift
precondition(index>0, "Index must be greater than zero.")
```
<br>

# References
- https://docs.swift.org/swift-book/
