// Copyright (c) 2026 Ojima Abraham. All rights reserved.
// Licensed under the Apache License, Version 2.0. See LICENSE file for details.

// Hand-written lexer. Consumes UTF-8 source one character at a time, emits
// typed tokens with byte spans. Newlines are significant (one instruction per
// line). Comments start with ; or # and run to end of line.

use crate::ast::Span;
use crate::diagnostics::AssemblerError;

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Identifier(String),
    Register(String),
    PredicateReg(String),
    SpecialReg(String),
    Integer(i64),
    Float(f64),
    Directive(String),
    Label(String),
    Predicate { register: String, negated: bool },
    Comma,
    Colon,
    Dot,
    Newline,
    Eof,
}

#[derive(Debug, Clone)]
pub struct SpannedToken {
    pub token: Token,
    pub span: Span,
}

pub struct Lexer<'a> {
    source: &'a str,
    chars: std::iter::Peekable<std::str::CharIndices<'a>>,
    position: usize,
}

impl<'a> Lexer<'a> {
    #[must_use]
    pub fn new(source: &'a str) -> Self {
        Self {
            source,
            chars: source.char_indices().peekable(),
            position: 0,
        }
    }

    pub fn tokenize(&mut self) -> Result<Vec<SpannedToken>, AssemblerError> {
        let mut tokens = Vec::new();

        loop {
            let token = self.next_token()?;
            let is_eof = token.token == Token::Eof;
            tokens.push(token);
            if is_eof {
                break;
            }
        }

        Ok(tokens)
    }

    fn next_token(&mut self) -> Result<SpannedToken, AssemblerError> {
        self.skip_whitespace_and_comments();

        let start = self.position;

        let Some((pos, ch)) = self.chars.next() else {
            return Ok(SpannedToken {
                token: Token::Eof,
                span: Span::new(start, start),
            });
        };

        self.position = pos + ch.len_utf8();

        let token = match ch {
            '\n' => Token::Newline,
            ',' => Token::Comma,
            ':' => Token::Colon,

            '.' if self.peek_char().is_some_and(|c| c.is_alphabetic()) => {
                self.scan_directive(start)?
            }

            '@' => self.scan_predicate(start)?,

            'r' if self.peek_char().is_some_and(|c| c.is_ascii_digit()) => {
                self.scan_register(start)
            }

            'p' if self.peek_char().is_some_and(|c| c.is_ascii_digit()) => {
                self.scan_predicate_register(start)
            }

            's' if self.peek_char() == Some('r') => {
                self.scan_special_register(start)?
            }

            '-' | '0'..='9' => self.scan_number(start, ch)?,

            _ if ch.is_alphabetic() || ch == '_' => {
                self.scan_identifier_or_label(start)
            }

            _ => {
                return Err(AssemblerError::UnexpectedCharacter {
                    char: ch,
                    span: Span::new(start, self.position),
                });
            }
        };

        Ok(SpannedToken {
            token,
            span: Span::new(start, self.position),
        })
    }

    fn peek_char(&mut self) -> Option<char> {
        self.chars.peek().map(|(_, c)| *c)
    }

    fn advance(&mut self) -> Option<char> {
        self.chars.next().map(|(pos, ch)| {
            self.position = pos + ch.len_utf8();
            ch
        })
    }

    fn skip_whitespace_and_comments(&mut self) {
        while let Some((pos, ch)) = self.chars.peek().copied() {
            match ch {
                ' ' | '\t' | '\r' => {
                    self.chars.next();
                    self.position = pos + 1;
                }
                ';' | '#' => {
                    self.chars.next();
                    self.position = pos + 1;
                    while let Some((p, c)) = self.chars.peek().copied() {
                        if c == '\n' {
                            break;
                        }
                        self.chars.next();
                        self.position = p + c.len_utf8();
                    }
                }
                _ => break,
            }
        }
    }

    fn scan_directive(&mut self, start: usize) -> Result<Token, AssemblerError> {
        let mut name = String::new();
        while let Some(ch) = self.peek_char() {
            if ch.is_alphanumeric() || ch == '_' {
                name.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        if name.is_empty() {
            return Err(AssemblerError::InvalidDirective {
                span: Span::new(start, self.position),
            });
        }

        Ok(Token::Directive(name))
    }

    fn scan_predicate(&mut self, start: usize) -> Result<Token, AssemblerError> {
        let negated = self.peek_char() == Some('!');
        if negated {
            self.advance();
        }

        if self.peek_char() != Some('p') {
            return Err(AssemblerError::InvalidPredicate {
                span: Span::new(start, self.position),
            });
        }
        self.advance();

        let mut num = String::new();
        while let Some(ch) = self.peek_char() {
            if ch.is_ascii_digit() {
                num.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        if num.is_empty() {
            return Err(AssemblerError::InvalidPredicate {
                span: Span::new(start, self.position),
            });
        }

        Ok(Token::Predicate {
            register: format!("p{num}"),
            negated,
        })
    }

    fn scan_register(&mut self, start: usize) -> Token {
        let mut reg = String::from("r");
        while let Some(ch) = self.peek_char() {
            if ch.is_ascii_digit() {
                reg.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        if self.peek_char() == Some('.') {
            self.advance();
            let mut suffix = String::new();
            while let Some(ch) = self.peek_char() {
                if ch.is_alphabetic() {
                    suffix.push(ch);
                    self.advance();
                } else {
                    break;
                }
            }
            reg.push('.');
            reg.push_str(&suffix);
        }

        let _ = start;
        Token::Register(reg)
    }

    fn scan_predicate_register(&mut self, _start: usize) -> Token {
        let mut reg = String::from("p");
        while let Some(ch) = self.peek_char() {
            if ch.is_ascii_digit() {
                reg.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        Token::PredicateReg(reg)
    }

    fn scan_special_register(&mut self, start: usize) -> Result<Token, AssemblerError> {
        let mut reg = String::from("s");
        while let Some(ch) = self.peek_char() {
            if ch.is_alphanumeric() || ch == '_' {
                reg.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        if !reg.starts_with("sr_") {
            return Err(AssemblerError::InvalidSpecialRegister {
                name: reg,
                span: Span::new(start, self.position),
            });
        }

        Ok(Token::SpecialReg(reg))
    }

    fn scan_number(&mut self, start: usize, first: char) -> Result<Token, AssemblerError> {
        let mut num_str = String::from(first);
        let mut is_hex = false;
        let mut is_float = false;

        if first == '0' && self.peek_char() == Some('x') {
            is_hex = true;
            num_str.push('x');
            self.advance();
        }

        while let Some(ch) = self.peek_char() {
            if is_hex {
                if ch.is_ascii_hexdigit() || ch == '_' {
                    if ch != '_' {
                        num_str.push(ch);
                    }
                    self.advance();
                } else {
                    break;
                }
            } else if ch.is_ascii_digit() || ch == '_' {
                if ch != '_' {
                    num_str.push(ch);
                }
                self.advance();
            } else if ch == '.' && !is_float {
                is_float = true;
                num_str.push(ch);
                self.advance();
            } else if (ch == 'e' || ch == 'E') && !is_hex {
                is_float = true;
                num_str.push(ch);
                self.advance();
                if self.peek_char() == Some('-') || self.peek_char() == Some('+') {
                    num_str.push(self.advance().unwrap());
                }
            } else {
                break;
            }
        }

        if is_float {
            num_str
                .parse::<f64>()
                .map(Token::Float)
                .map_err(|_| AssemblerError::InvalidNumber {
                    span: Span::new(start, self.position),
                })
        } else if is_hex {
            let hex_str = num_str.trim_start_matches("0x");
            i64::from_str_radix(hex_str, 16)
                .map(Token::Integer)
                .map_err(|_| AssemblerError::InvalidNumber {
                    span: Span::new(start, self.position),
                })
        } else {
            num_str
                .parse::<i64>()
                .map(Token::Integer)
                .map_err(|_| AssemblerError::InvalidNumber {
                    span: Span::new(start, self.position),
                })
        }
    }

    fn scan_identifier_or_label(&mut self, _start: usize) -> Token {
        let ident_start = self.position - 1;
        while let Some(ch) = self.peek_char() {
            if ch.is_alphanumeric() || ch == '_' {
                self.advance();
            } else {
                break;
            }
        }

        let ident = &self.source[ident_start..self.position];

        self.skip_whitespace_and_comments();
        if self.peek_char() == Some(':') {
            Token::Label(ident.to_string())
        } else {
            Token::Identifier(ident.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenize(source: &str) -> Vec<Token> {
        Lexer::new(source)
            .tokenize()
            .unwrap()
            .into_iter()
            .map(|st| st.token)
            .collect()
    }

    #[test]
    fn test_lexer_simple_instruction() {
        let tokens = tokenize("iadd r0, r1, r2");
        assert_eq!(
            tokens,
            vec![
                Token::Identifier("iadd".into()),
                Token::Register("r0".into()),
                Token::Comma,
                Token::Register("r1".into()),
                Token::Comma,
                Token::Register("r2".into()),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_lexer_predicate() {
        let tokens = tokenize("@p0 iadd r0, r1, r2");
        assert_eq!(
            tokens,
            vec![
                Token::Predicate { register: "p0".into(), negated: false },
                Token::Identifier("iadd".into()),
                Token::Register("r0".into()),
                Token::Comma,
                Token::Register("r1".into()),
                Token::Comma,
                Token::Register("r2".into()),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_lexer_negated_predicate() {
        let tokens = tokenize("@!p1 fadd r0, r1, r2");
        assert_eq!(
            tokens,
            vec![
                Token::Predicate { register: "p1".into(), negated: true },
                Token::Identifier("fadd".into()),
                Token::Register("r0".into()),
                Token::Comma,
                Token::Register("r1".into()),
                Token::Comma,
                Token::Register("r2".into()),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_lexer_directive() {
        let tokens = tokenize(".kernel my_kernel");
        assert_eq!(
            tokens,
            vec![
                Token::Directive("kernel".into()),
                Token::Identifier("my_kernel".into()),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_lexer_label() {
        let tokens = tokenize("loop_start:");
        assert_eq!(
            tokens,
            vec![
                Token::Label("loop_start".into()),
                Token::Colon,
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_lexer_hex_immediate() {
        let tokens = tokenize("mov_imm r0, 0xFF");
        assert_eq!(
            tokens,
            vec![
                Token::Identifier("mov_imm".into()),
                Token::Register("r0".into()),
                Token::Comma,
                Token::Integer(255),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_lexer_negative_immediate() {
        let tokens = tokenize("mov_imm r0, -42");
        assert_eq!(
            tokens,
            vec![
                Token::Identifier("mov_imm".into()),
                Token::Register("r0".into()),
                Token::Comma,
                Token::Integer(-42),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_lexer_float_immediate() {
        let tokens = tokenize("mov_imm r0, 3.14");
        assert_eq!(
            tokens,
            vec![
                Token::Identifier("mov_imm".into()),
                Token::Register("r0".into()),
                Token::Comma,
                Token::Float(3.14),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_lexer_special_register() {
        let tokens = tokenize("mov r0, sr_thread_id_x");
        assert_eq!(
            tokens,
            vec![
                Token::Identifier("mov".into()),
                Token::Register("r0".into()),
                Token::Comma,
                Token::SpecialReg("sr_thread_id_x".into()),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_lexer_register_half() {
        let tokens = tokenize("hadd r0.lo, r1.hi, r2.lo");
        assert_eq!(
            tokens,
            vec![
                Token::Identifier("hadd".into()),
                Token::Register("r0.lo".into()),
                Token::Comma,
                Token::Register("r1.hi".into()),
                Token::Comma,
                Token::Register("r2.lo".into()),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_lexer_comment() {
        let tokens = tokenize("iadd r0, r1, r2 ; this is a comment");
        assert_eq!(
            tokens,
            vec![
                Token::Identifier("iadd".into()),
                Token::Register("r0".into()),
                Token::Comma,
                Token::Register("r1".into()),
                Token::Comma,
                Token::Register("r2".into()),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_lexer_multiple_lines() {
        let tokens = tokenize("iadd r0, r1, r2\nisub r3, r4, r5");
        assert_eq!(
            tokens,
            vec![
                Token::Identifier("iadd".into()),
                Token::Register("r0".into()),
                Token::Comma,
                Token::Register("r1".into()),
                Token::Comma,
                Token::Register("r2".into()),
                Token::Newline,
                Token::Identifier("isub".into()),
                Token::Register("r3".into()),
                Token::Comma,
                Token::Register("r4".into()),
                Token::Comma,
                Token::Register("r5".into()),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_lexer_predicate_register_operand() {
        let tokens = tokenize("select r0, p1, r1, r2");
        assert_eq!(
            tokens,
            vec![
                Token::Identifier("select".into()),
                Token::Register("r0".into()),
                Token::Comma,
                Token::PredicateReg("p1".into()),
                Token::Comma,
                Token::Register("r1".into()),
                Token::Comma,
                Token::Register("r2".into()),
                Token::Eof,
            ]
        );
    }
}
