#pragma once
#include <SFML/Graphics.hpp>

namespace Col {
    inline const sf::Color Bg       {18,  14,   6};
    inline const sf::Color Panel    {28,  20,   8};
    inline const sf::Color Panel2   {38,  28,  12};
    inline const sf::Color Border   {80,  55,  22};
    inline const sf::Color BorderHi {160, 110,  40};
    inline const sf::Color Gold     {200, 145,  40};
    inline const sf::Color GoldLt   {240, 192,  80};
    inline const sf::Color Text     {232, 213, 163};
    inline const sf::Color Muted    {130, 112,  80};
    inline const sf::Color Red      {176,  48,  48};
    inline const sf::Color RedLt    {220,  80,  80};
    inline const sf::Color Green    { 60, 130,  55};
    inline const sf::Color GreenLt  { 90, 180,  80};
    inline const sf::Color Blue     { 42,  85, 138};
    inline const sf::Color BlueLt   { 80, 140, 200};
    inline const sf::Color Purple   {100,  55, 155};
    inline const sf::Color PurpleLt {150,  90, 210};
    inline const sf::Color Injury   {200,  80,  40};
    inline const sf::Color Overlay  {  0,   0,   0, 190};
}

constexpr int WIN_W    = 1280;
constexpr int WIN_H    = 800;
constexpr int HEADER_H = 62;
constexpr int LOG_H    = 88;
constexpr int LEFT_W   = 295;
constexpr int MID_X    = LEFT_W;
constexpr int MID_W    = WIN_W - LEFT_W;
constexpr int BODY_H   = WIN_H - HEADER_H - LOG_H;
constexpr int PAD      = 8;

