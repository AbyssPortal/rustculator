mod expression_math {
    use std::collections::HashMap;
    use std::fmt;

    // enum MathFunctionError {
    //     IndifferentiableError(String),
    //     IncorrectParamsError(String),
    // }

    // pub trait MathFunction {
    //     fn calculate(params: Vec<Expression>) -> Result<Expression, MathFunctionError>;

    //     fn derive(params: Vec<Expression>, index: usize) -> Result<Expression, MathFunctionError>;
    // }

    #[test]
    fn test_calc() {
        let exprs = [
            ("1+2", 3),
            ("3*4", 12),
            ("10-5", 5),
            ("8/2", 4),
            ("2+3*4", 14),
            ("(2+3)*4", 20),
            ("2*(3+4)", 14),
            ("10/2-3", 2),
            ("5+6-7*8/2", -17),
            ("-1", -1),
            ("-(1+2)", -3),
            ("-(3*4)", -12),
            ("-(10-5)", -5),
            ("-(8/2)", -4),
            ("10+-1", 9),
            ("+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-1", 1),
            ("-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-1", -1),
            ("---1", -1)
        ];
        let mut okay = true;
        for (expr, expected) in exprs {
            println!("====TEST====\n {} == {}", expr, expected);
            let calc = calculate_expression(
                parse_expression(expr).expect("cant parse expr"),
                &HashMap::<String, Number>::new(),
            )
            .expect("cant calculate");
            if calc != expected {
                println!("\x1b[31m====FAIL====\x1b[0m");
                println!("====RECIEVED====");
                println!("{} == {} ", expr, calc);
                okay = false;
            } else {
                println!("\x1b[32m====GOOD=====\x1b[0m");
            }
        }
        if !okay {
            panic!();
        }
    }

    #[test]
    fn test_variables() {
        {
            let vars: HashMap<String, Number> = [("x".to_string(), 2)].iter().cloned().collect();

            println!("{:?}", parse_expression("x+2"));

            assert_eq!(
                calculate_expression(parse_expression("x+2").unwrap(), &vars).expect("cant calc"),
                4
            );
        }
        {
            let vars: HashMap<String, Number> = [("x".to_string(), 2), ("y".to_string(), 1)]
                .iter()
                .cloned()
                .collect();

            assert_eq!(
                calculate_expression(parse_expression("x*(y+2)").unwrap(), &vars)
                    .expect("cant calc"),
                6
            );
        }
    }

    #[test]
    fn test_derivative() {
        let original = parse_expression("x*x*x").unwrap();
        println!("d/dx({}) = {}", original.clone(), derivative(original, "x"));
    }
    fn parse_tokens(input: &[Token]) -> Result<Expression, ExpressionError> {
        if input.len() == 0 {
            return Err(ExpressionError::ExpectedTokenError);
        } else if input.len() == 1 {
            return match &input[0] {
                Token::Operator(_) => Err(ExpressionError::UnexpectedTokenError(input[0].clone())),
                Token::Expression(expr) => Ok(expr.clone()),
                Token::Constant(num) => Ok(Expression::Constant(num.clone())),
                Token::Variable(name) => Ok(Expression::Variable(name.clone())),
            };
        };

        // try the unary.
        const UNARY_OPERATORS: &[Operator] = &[Operator::Addition, Operator::Subtraction];
        match input[0] {
            Token::Operator(op) => {
                if UNARY_OPERATORS.contains(&op) {
                    return Ok(Expression::UnaryOperation(
                        op,
                        Box::new(parse_tokens(&input[1..])?),
                    ));
                }
            }
            _ => {}
        }

        const OPERATORS_BY_PRIORITY: &[&[Operator]] = &[
            &[Operator::Subtraction, Operator::Addition], // First priority operators
            &[Operator::Division, Operator::Multiplication], // Second priority operators
        ];
        //^it's reversed. THIS IS INTENTIONAL^

        for operators in OPERATORS_BY_PRIORITY {
            let tokens = input.into_iter().enumerate().rev();
            for (index, token) in tokens {
                match token {
                    Token::Operator(some_op) => {
                        if operators.contains(&some_op) {
                            match parse_tokens(&input[0..index]) {
                                Ok(expr1) => match parse_tokens(&input[(index + 1)..]) {
                                    Ok(expr2) => {
                                        return Ok(Expression::BinaryOperation(
                                            Box::new(expr1),
                                            (some_op).clone(),
                                            Box::new(expr2),
                                        ));
                                    }
                                    Err(_) => {
                                        continue;
                                    }
                                },
                                Err(_) => {
                                    continue;
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        Err(ExpressionError::BadFormatError)
    }

    fn calculate_expression(
        expr: Expression,
        variables: &HashMap<String, Number>,
    ) -> Result<Number, ExpressionError> {
        Ok(match expr {
            Expression::Constant(num) => num,
            Expression::BinaryOperation(expr1, op, expr2) => {
                let num1 = calculate_expression(*expr1, variables)?;
                let num2 = calculate_expression(*expr2, variables)?;
                match op {
                    Operator::Addition => num1 + num2,
                    Operator::Subtraction => num1 - num2,
                    Operator::Multiplication => num1 * num2,
                    Operator::Division => num1 / num2,
                }
            }
            Expression::Variable(name) => {
                if let Some(value) = variables.get(&name) {
                    *value
                } else {
                    return Err(ExpressionError::UnexpectedVariableError(name));
                }
            }
            Expression::UnaryOperation(op, expr1) => match op {
                Operator::Addition => calculate_expression(*expr1, variables)?,
                Operator::Subtraction => -calculate_expression(*expr1, variables)?,
                _ => {
                    return Err(ExpressionError::InternalError);
                }
            },
        })
    }

    pub fn derivative(expr: Expression, variable: &str) -> Expression {
        match expr {
            Expression::Constant(_) => Expression::Constant(0),
            Expression::UnaryOperation(_, _) => {
                todo!()
            }
            Expression::Variable(some_var) => {
                if some_var == variable {
                    Expression::Constant(1)
                } else {
                    Expression::Constant(0)
                }
            }
            Expression::BinaryOperation(expr1, op, expr2) => match op {
                Operator::Addition => Expression::BinaryOperation(
                    Box::new(derivative(*expr1, variable)),
                    Operator::Addition,
                    Box::new(derivative(*expr2, variable)),
                ),
                Operator::Subtraction => Expression::BinaryOperation(
                    Box::new(derivative(*expr1, variable)),
                    Operator::Subtraction,
                    Box::new(derivative(*expr2, variable)),
                ),
                Operator::Multiplication => Expression::BinaryOperation(
                    Box::new(Expression::BinaryOperation(
                        Box::new(derivative(*expr1.clone(), variable)),
                        Operator::Multiplication,
                        Box::new(*expr2.clone()),
                    )),
                    Operator::Addition,
                    Box::new(Expression::BinaryOperation(
                        Box::new(derivative(*expr2, variable)),
                        Operator::Multiplication,
                        Box::new(*expr1),
                    )),
                ),
                Operator::Division => Expression::BinaryOperation(
                    Box::new(Expression::BinaryOperation(
                        Box::new(Expression::BinaryOperation(
                            Box::new(derivative(*expr1.clone(), variable)),
                            Operator::Multiplication,
                            Box::new(*expr2.clone()),
                        )),
                        Operator::Subtraction,
                        Box::new(Expression::BinaryOperation(
                            Box::new(derivative(*expr2.clone(), variable)),
                            Operator::Multiplication,
                            Box::new(*expr1),
                        )),
                    )),
                    Operator::Division,
                    Box::new(Expression::BinaryOperation(
                        Box::new(*expr2.clone()),
                        Operator::Multiplication,
                        Box::new(*expr2),
                    )),
                ),
            },
        }
    }

    pub fn parse_expression(input: &str) -> Result<Expression, ExpressionError> {
        let mut tokens = Vec::new();
        let mut chars = input.chars().enumerate().peekable();
        while let Some((index, letter)) = chars.next() {
            // dosen't work for weird chars. i dont give a shittE
            if letter.is_whitespace() {
                continue;
            } else if letter == '(' {
                let start_index = index;
                let mut brack_count = 1;
                if (loop {
                    if let Some((index, letter)) = chars.next() {
                        if letter == ')' {
                            brack_count -= 1;
                            if brack_count == 0 {
                                tokens.push(Token::Expression(parse_expression(
                                    &input[(start_index + 1)..index],
                                )?));
                                break true;
                            }
                        } else if letter == '(' {
                            brack_count += 1;
                        }
                    } else {
                        break false;
                    }
                } == false)
                {
                    return Err(ExpressionError::MissingCharacterError(')'));
                }
            } else if letter.is_alphabetic() {
                let mut name: String = letter.to_string();
                while let Some((_index, letter)) = chars.peek() {
                    if letter.is_alphabetic() {
                        name.push(*letter);
                        chars.next();
                    } else {
                        break;
                    }
                }

                tokens.push(Token::Variable(name));
            } else if letter.is_numeric() {
                let start_index = index;
                let mut end_index = start_index;
                while let Some((index, letter)) = chars.peek() {
                    if letter.is_alphanumeric() {
                        end_index = *index;
                        chars.next();
                    } else {
                        break;
                    }
                }
                let num = number_from_string(&input[start_index..(end_index + 1)])?;
                tokens.push(Token::Constant(num));
            } else if let Some((op, consumed)) = starts_with_operator(&input[index..]) {
                tokens.push(Token::Operator(op));
                for _ in 0..(consumed - 1) {
                    chars.next();
                }
            } else {
                return Err(ExpressionError::UnexpectedCharacterError(letter));
            }
        }
        parse_tokens(&tokens)
    }

    fn starts_with_operator(input: &str) -> Option<(Operator, usize)> {
        if input.starts_with("+") {
            Some((Operator::Addition, 1))
        } else if input.starts_with("-") {
            Some((Operator::Subtraction, 1))
        } else if input.starts_with("*") {
            Some((Operator::Multiplication, 1))
        } else if input.starts_with("/") {
            Some((Operator::Division, 1))
        } else {
            None
        }
    }

    fn number_from_string(name: &str) -> Result<Number, ExpressionError> {
        match name.parse::<Number>() {
            Ok(num) => Ok(num),
            Err(_) => Err(ExpressionError::BadNumberFormatError),
        }
    }

    #[derive(Debug, Clone)]
    pub enum Token {
        Operator(Operator),
        Variable(String),
        Expression(Expression),
        Constant(Number),
    }
    #[derive(Debug, Clone)]
    pub enum ExpressionError {
        UnexpectedTokenError(Token),
        ExpectedTokenError,
        UnexpectedCharacterError(char),
        MissingCharacterError(char),
        InternalError,
        BadNumberFormatError,
        UnexpectedVariableError(String),
        BadFormatError,
    }
    type Number = i32;
    #[derive(Debug, Clone, PartialEq, Eq, Copy)]
    pub enum Operator {
        Addition,
        Subtraction,
        Multiplication,
        Division,
    }

    impl fmt::Display for Operator {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            let symbol = match self {
                Operator::Addition => "+",
                Operator::Subtraction => "-",
                Operator::Multiplication => "*",
                Operator::Division => "/",
            };
            write!(f, "{}", symbol)
        }
    }


    
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum Expression {
        Constant(Number),
        BinaryOperation(Box<Expression>, Operator, Box<Expression>),
        UnaryOperation(Operator, Box<Expression>),
        Variable(String),
    }

    fn is_const(expr: &Expression) -> bool {
        match expr {
            Expression::Variable(_) => false,
            Expression::BinaryOperation(expr1, _, expr2) => is_const(expr1) && is_const(expr2),
            Expression::Constant(_) => true,
            Expression::UnaryOperation(_, expr1) => is_const(&expr1),
        }
    }

    pub fn simplify_expression(mut expr: Expression) -> Expression {
        expr = simplify_recursive_wrapper(expr);

        if is_const(&expr) {
            Expression::Constant(
                calculate_expression(expr, &HashMap::<String, Number>::new()).unwrap(),
            )
        } else {
            expr
        }
    }

    //RETURNS: the gcd, mul1 where gcd*mul1 = expr1, mul2. not a fancy scientifc gcd algorithm
    fn expression_gcd(
        expr1: &Expression,
        expr2: &Expression,
    ) -> (Expression, Expression, Expression) {
        if expr1 == expr2 {
            return (
                expr1.clone(),
                Expression::Constant(1),
                Expression::Constant(1),
            );
        }

        match (expr1, expr2) {
            (
                Expression::BinaryOperation(lhs1, Operator::Multiplication, rhs1),
                Expression::BinaryOperation(lhs2, Operator::Multiplication, rhs2),
            ) => {
                if lhs1 == lhs2 {
                    (*lhs1.clone(), *rhs1.clone(), *rhs2.clone())
                } else if lhs1 == rhs2 {
                    (*lhs1.clone(), *rhs1.clone(), *lhs2.clone())
                } else if rhs1 == lhs2 {
                    (*rhs1.clone(), *lhs1.clone(), *rhs2.clone())
                } else if rhs1 == rhs2 {
                    (*rhs1.clone(), *lhs1.clone(), *lhs2.clone())
                } else {
                    (Expression::Constant(1), expr1.clone(), expr2.clone())
                }
            }
            (Expression::BinaryOperation(lhs1, Operator::Multiplication, rhs1), expr2) => {
                if **lhs1 == *expr2 {
                    (*lhs1.clone(), *rhs1.clone(), Expression::Constant(1))
                } else if **rhs1 == *expr2 {
                    (*rhs1.clone(), *lhs1.clone(), Expression::Constant(1))
                } else {
                    (Expression::Constant(1), expr1.clone(), expr2.clone())
                }
            }
            (expr1, Expression::BinaryOperation(lhs2, Operator::Multiplication, rhs2)) => {
                if **lhs2 == *expr1 {
                    (*lhs2.clone(), Expression::Constant(1), *rhs2.clone())
                } else if **rhs2 == *expr1 {
                    (*rhs2.clone(), Expression::Constant(1), *lhs2.clone())
                } else {
                    (Expression::Constant(1), expr1.clone(), expr2.clone())
                }
            }
            _ => (Expression::Constant(1), expr1.clone(), expr2.clone()),
        }
    }

    fn simplify_recursive_wrapper(expr: Expression) -> Expression {
        static mut DEPTH: i32 = 0;
        unsafe {
            // it's safe this is thread safety bullshit
            DEPTH += 1;
            let prefix = " ".repeat((4 * DEPTH) as usize);
            let print = false;
            if print {
                println!("{}given: {}", prefix, expr);
            }

            let res = simplify_recursive_internal(expr);
            if print {
                println!("{}calculated: {}", prefix, res);
            }
            DEPTH -= 1;
            res
        }
    }

    fn simplify_recursive_internal(expr: Expression) -> Expression {
        match &expr {
            Expression::Constant(_) => expr,
            Expression::UnaryOperation(_, _) => {
                todo!()
            }
            Expression::Variable(_) => expr,
            Expression::BinaryOperation(lhs, op, rhs) => match op {
                Operator::Subtraction => {
                    let simple_lhs = simplify_expression(*lhs.clone());
                    let simple_rhs = simplify_expression(*rhs.clone());
                    if simple_rhs == Expression::Constant(0) {
                        simple_lhs
                    } else if let Some(simplifed) = simplify_gcd(&simple_lhs, &simple_rhs, *op) {
                        simplifed
                    } else {
                        Expression::BinaryOperation(Box::new(simple_lhs), *op, Box::new(simple_rhs))
                    }
                }
                Operator::Addition => {
                    let simple_lhs = simplify_expression(*lhs.clone());
                    let simple_rhs = simplify_expression(*rhs.clone());
                    if simple_lhs == Expression::Constant(0) {
                        simple_rhs
                    } else if simple_rhs == Expression::Constant(0) {
                        simple_lhs
                    } else if let Some(simplifed) = simplify_gcd(&simple_lhs, &simple_rhs, *op) {
                        simplifed
                    } else {
                        Expression::BinaryOperation(Box::new(simple_lhs), *op, Box::new(simple_rhs))
                    }
                }
                Operator::Multiplication => {
                    let simple_lhs = simplify_expression(*lhs.clone());
                    let simple_rhs = simplify_expression(*rhs.clone());
                    if simple_lhs == Expression::Constant(0)
                        || simple_rhs == Expression::Constant(0)
                    {
                        Expression::Constant(0)
                    } else if simple_lhs == Expression::Constant(1) {
                        simple_rhs
                    } else if simple_rhs == Expression::Constant(1) {
                        simple_lhs
                    } else if matches!(simple_lhs, Expression::Variable(_)) {
                        Expression::BinaryOperation(Box::new(simple_rhs), *op, Box::new(simple_lhs))
                    } else {
                        Expression::BinaryOperation(Box::new(simple_lhs), *op, Box::new(simple_rhs))
                    }
                }
                _ => expr,
            },
        }
    }

    //op is either + or - generally
    fn simplify_gcd(
        simple_lhs: &Expression,
        simple_rhs: &Expression,
        op: Operator,
    ) -> Option<Expression> {
        let gcd = expression_gcd(&simple_lhs, &simple_rhs);
        if gcd.0 != Expression::Constant(1) {
            Some(simplify_expression(Expression::BinaryOperation(
                Box::new(gcd.0),
                Operator::Multiplication,
                Box::new(simplify_expression(Expression::BinaryOperation(
                    Box::new(gcd.1),
                    op,
                    Box::new(gcd.2),
                ))),
            )))
        } else {
            None
        }
    }

    fn operator_priority(op: &Operator) -> usize {
        match op {
            Operator::Multiplication => 2,
            Operator::Division => 2,
            Operator::Addition => 1,

            Operator::Subtraction => 1,
        }
    }

    fn is_associative(op: Operator) -> bool {
        match op {
            Operator::Addition => true,
            Operator::Subtraction => false,
            Operator::Multiplication => true,
            Operator::Division => false,
        }
    }

    impl fmt::Display for Expression {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            Ok(match self {
                Self::BinaryOperation(expr1, op, expr2) => {
                    let lhs_needs_bracket = match &**expr1 {
                        Self::BinaryOperation(_, sub_op, _) => {
                            operator_priority(sub_op) < operator_priority(op) //associative default
                        }
                        _ => false,
                    };
                    let rhs_needs_bracket = match &**expr2 {
                        Self::BinaryOperation(_, sub_op, _) => {
                            operator_priority(sub_op) <= operator_priority(op)
                                && (!(sub_op == op && is_associative(*op))) //associative non default
                        }
                        _ => false,
                    };
                    if lhs_needs_bracket {
                        write!(f, "({}) ", expr1)?;
                    } else {
                        write!(f, "{} ", expr1)?;
                    }
                    write!(f, "{}", op)?;
                    if rhs_needs_bracket {
                        write!(f, " ({})", expr2)?;
                    } else {
                        write!(f, " {}", expr2)?;
                    }
                }
                Self::Constant(num) => {
                    write!(f, "{}", num)?;
                }
                Self::UnaryOperation(op, expr) => {
                    write!(f, "{}({})", op, expr)?;
                }
                Self::Variable(name) => {
                    write!(f, "{}", name)?;
                }
            })
        }
    }
}

use crate::expression_math::*;
use std::io;

fn main() {
    let mut input = String::new();
    loop {
        match io::stdin().read_line(&mut input) {
            Ok(_) => {
                if input == "q" || input == "quit" {
                    return;
                }
                match parse_expression(&input.trim()) {
                    Ok(expr) => {
                        println!("read: {}, {:?}", expr, expr);
                        let simplifed = simplify_expression(expr);
                        println!("simplified: {}", simplifed);
                        let derivative = simplify_expression(derivative(simplifed, "x"));
                        println!("derivative: {}", derivative);
                    }
                    Err(e) => {
                        eprintln!("Error parsing expression: {:?}", e);
                    }
                }
            }
            Err(error) => {
                eprintln!("Error reading input: {}", error);
            }
        }
        input.clear();
    }
}
