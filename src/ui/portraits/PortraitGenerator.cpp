#include "PortraitGenerator.hpp"
#include "../../enum/enums.hpp"

#include <SFML/Graphics/CircleShape.hpp>
#include <SFML/Graphics/ConvexShape.hpp>
#include <SFML/Graphics/RectangleShape.hpp>


sf::Texture PortraitGenerator::generate(unsigned seed, Race race, Gender gender) {
    const auto context = createContext(seed, race, gender);

    std::vector<std::unique_ptr<PortraitLayer>> layers;
    layers.push_back(createNeckLayer(context));
    layers.push_back(createHairLayer(context));
    layers.push_back(createSkinLayer(context));
    layers.push_back(createEarLayer(context));
    layers.push_back(createEyebrowLayer(context));
    layers.push_back(createEyeLayer(context));
    layers.push_back(createNoseLayer(context));
    layers.push_back(createMouthLayer(context));
    layers.push_back(createFacialHairLayer(context));
    layers.push_back(createScarLayer(context));
    return combineLayers(layers);
}

sf::Texture PortraitGenerator::combineLayers(const std::vector<std::unique_ptr<PortraitLayer>>& layers) {
    sf::RenderTexture rt;
    rt.create(120, 128);
    rt.clear(sf::Color::Transparent);
    for (const auto& layer : layers)
        layer->draw(rt);
    rt.display();
    return rt.getTexture();
}

std::unique_ptr<PortraitLayer> PortraitGenerator::createNeckLayer(const PortraitContext context) {
    auto layer = std::make_unique<PortraitLayer>();
    layer->rt.create(120,128);
    layer->rt.clear(sf::Color::Transparent);

    float cx = 60.f, cy = 56.f;

    sf::RectangleShape neck({18.f, 22.f});
    neck.setOrigin(9.f, 0.f);
    neck.setFillColor(context.skin);
    neck.setPosition(cx, cy + context.fh * 0.5f - 6.f);

    layer->rt.draw(neck);
    layer->rt.display();
    return layer;
}

std::unique_ptr<PortraitLayer> PortraitGenerator::createHairLayer(const PortraitContext context) {
    auto layer = std::make_unique<PortraitLayer>();
    layer->rt.create(120,128);
    layer->rt.clear(sf::Color::Transparent);

    float cx = 60.f, cy = 56.f;

    if (context.hairStyle != 3) {
        float hr = context.fr + 4.f;
        float hh = context.fh * 0.5f + 6.f;

        ellipse(layer->rt, cx, cy - 4.f, hr, hh, context.hair);

        if (context.hairStyle == 2) {
            for (int s : {-1,1}) {
                sf::RectangleShape lock({11.f, 42.f});
                lock.setFillColor(context.hair);
                lock.setPosition(cx + s*(context.fr + (s<0 ? -11.f : 1.f)), cy-8.f);
                layer->rt.draw(lock);
            }
        }

        if (context.hairStyle == 4) {
            ellipse(layer->rt, cx, cy - 6.f, context.fr + 8.f, context.fh * 0.5f + 8.f, context.hair);
        }
    }

    layer->rt.display();
    return layer;
}

std::unique_ptr<PortraitLayer> PortraitGenerator::createSkinLayer(PortraitContext context) {
    auto layer = std::make_unique<PortraitLayer>();
    layer->rt.create(120,128);
    layer->rt.clear(sf::Color::Transparent);

    float cx = 60.f, cy = 56.f;

    sf::CircleShape face(context.fr);
    face.setOrigin(context.fr, context.fr);
    face.setScale(1.f, context.fh * 0.5f / context.fr);
    face.setPosition(cx, cy);
    face.setFillColor(context.skin);

    layer->rt.draw(face);

    if (context.race == Race::Orc) {
        sf::CircleShape shadow(context.fr * 0.85f);
        shadow.setOrigin(context.fr * 0.85f, context.fr * 0.85f);
        shadow.setScale(1.f, context.fh * 0.09f / (context.fr * 0.85f));
        shadow.setPosition(cx, cy - context.fh * 0.14f);
        shadow.setFillColor(context.shadow);
        layer->rt.draw(shadow);
    }

    layer->rt.display();
    return layer;
}

std::unique_ptr<PortraitLayer> PortraitGenerator::createFacialHairLayer(const PortraitContext context) {
    auto layer = std::make_unique<PortraitLayer>();
    layer->rt.create(120,128);
    layer->rt.clear(sf::Color::Transparent);

    if (context.fhType == 0) return layer;

    float cx = 60.f, cy = 56.f;
    float my = cy + context.fh * 0.28f;

    if (context.fhType == 2) {
        sf::RectangleShape mus({18.f,5.f});
        mus.setOrigin(9.f,2.5f);
        mus.setFillColor(context.hair);
        mus.setPosition(cx,my-4.f);
        layer->rt.draw(mus);
    } else {
        float beardH = (context.race==Race::Dwarf) ? 18.f : 14.f;
        ellipse(layer->rt, cx, my+beardH, 18.f, beardH, context.hair);
    }

    layer->rt.display();
    return layer;
}

std::unique_ptr<PortraitLayer> PortraitGenerator::createEyeLayer(const PortraitContext context) {
    auto layer = std::make_unique<PortraitLayer>();
    layer->rt.create(120,128);
    layer->rt.clear(sf::Color::Transparent);

    float cx = 60.f, cy = 56.f;
    float ey = cy - context.fh * 0.06f;

    float eyeRx = (context.race == Race::Elf) ? 7.5f : 7.f;
    float eyeRy = (context.race == Race::Elf) ? 5.5f : 6.f;

    for (int s : {-1,1}) {
        float ex = cx + s * 14.f;

        ellipse(layer->rt, ex, ey, eyeRx, eyeRy, sf::Color::White);
        ellipse(layer->rt, ex, ey, 4.5f, 4.5f, context.iris);
        ellipse(layer->rt, ex, ey, 2.f, 2.f, sf::Color::Black);

        sf::CircleShape hl(1.5f);
        hl.setFillColor({255,255,255,200});
        hl.setPosition(ex+1.f, ey-3.5f);
        layer->rt.draw(hl);
    }

    layer->rt.display();
    return layer;
}

std::unique_ptr<PortraitLayer> PortraitGenerator::createEarLayer(const PortraitContext context) {
    auto layer = std::make_unique<PortraitLayer>();
    layer->rt.create(120,128);
    layer->rt.clear(sf::Color::Transparent);

    float cx = 60.f, cy = 56.f;

    if (context.race == Race::Elf) {
        for (int s : {-1,1}) {
            sf::ConvexShape ear(3);
            ear.setFillColor(context.skin);

            float ex = cx + s*(context.fr + 2.f);
            ear.setPoint(0, {ex, cy+4.f});
            ear.setPoint(1, {ex, cy-6.f});
            ear.setPoint(2, {ex + s*14.f, cy-14.f});

            layer->rt.draw(ear);
        }
    } else {
        ellipse(layer->rt, cx - context.fr - 4.f, cy, 5.f, 6.f, context.skin);
        ellipse(layer->rt, cx + context.fr + 4.f, cy, 5.f, 6.f, context.skin);
    }

    layer->rt.display();
    return layer;
}

std::unique_ptr<PortraitLayer> PortraitGenerator::createEyebrowLayer(const PortraitContext context) {
    auto layer = std::make_unique<PortraitLayer>();
    layer->rt.create(120,128);
    layer->rt.clear(sf::Color::Transparent);

    float cx = 60.f, cy = 56.f;

    float browY = cy - context.fh * 0.16f;
    float browH = (context.race==Race::Dwarf || context.race==Race::Orc) ? 5.f : 3.f;
    float browW = (context.race==Race::Dwarf) ? 16.f : 13.f;

    for (int s : {-1,1}) {
        sf::RectangleShape b({browW, browH});
        b.setFillColor(context.brow);
        b.setPosition(cx - s*(s<0 ? -9.f : browW+9.f), browY);
        layer->rt.draw(b);
    }

    layer->rt.display();
    return layer;
}

std::unique_ptr<PortraitLayer> PortraitGenerator::createNoseLayer(const PortraitContext context) {
    auto layer = std::make_unique<PortraitLayer>();
    layer->rt.create(120,128);
    layer->rt.clear(sf::Color::Transparent);

    float cx = 60.f, cy = 56.f;

    const float ny = cy + context.fh * 0.10f;
    const float nrx = (context.race == Race::Dwarf) ? 4.f : 3.f;
    const float nry = (context.race == Race::Dwarf) ? 3.f : 2.5f;

    ellipse(layer->rt, cx-nrx-1.5f, ny, nrx, nry, context.shadow);
    ellipse(layer->rt, cx+nrx+1.5f, ny, nrx, nry, context.shadow);

    layer->rt.display();
    return layer;
}

std::unique_ptr<PortraitLayer> PortraitGenerator::createMouthLayer(const PortraitContext context) {
    auto layer = std::make_unique<PortraitLayer>();
    layer->rt.create(120,128);
    layer->rt.clear(sf::Color::Transparent);

    float cx = 60.f, cy = 56.f;
    float my = cy + context.fh * 0.30f;

    sf::RectangleShape upper({16.f,3.f});
    upper.setOrigin(8.f,1.5f);
    upper.setFillColor(context.lip);
    upper.setPosition(cx,my);
    layer->rt.draw(upper);

    sf::RectangleShape lower({13.f,4.f});
    lower.setOrigin(6.5f,0.f);
    lower.setFillColor(context.lip);
    lower.setPosition(cx,my+3.f);
    layer->rt.draw(lower);

    if (context.race == Race::Orc) {
        sf::Color tusk{225,215,190};
        for (int s : {-1,1}) {
            sf::RectangleShape t({4.f,9.f});
            t.setFillColor(tusk);
            t.setPosition(cx+s*(s<0?10.f:-4.f)-4.f, my+5.f);
            layer->rt.draw(t);
        }
    }

    layer->rt.display();
    return layer;
}

std::unique_ptr<PortraitLayer> PortraitGenerator::createScarLayer(PortraitContext context) {
    auto layer = std::make_unique<PortraitLayer>();
    layer->rt.create(120,128);
    layer->rt.clear(sf::Color::Transparent);

    auto ri = [&](int lo, int hi) {
        return std::uniform_int_distribution<int>(lo, hi)(context.prng);
    };

    if (ri(0,6) != 0) return layer;

    float cx = 60.f, cy = 56.f;

    sf::Color sc = {
        static_cast<sf::Uint8>(context.skin.r * 0.6f),
        static_cast<sf::Uint8>(context.skin.g * 0.5f),
        static_cast<sf::Uint8>(context.skin.b * 0.5f)
    };

    sf::RectangleShape scar({2.5f, static_cast<float>(ri(14,22))});
    scar.setFillColor(sc);
    scar.setOrigin(1.25f,0.f);
    scar.setRotation(static_cast<float>(ri(-25,25)));
    scar.setPosition(cx + ri(-12,12), cy - context.fh * 0.25f);

    layer->rt.draw(scar);
    layer->rt.display();
    return layer;
}

void PortraitGenerator::ellipse(sf::RenderTexture& rt, float x, float y, float rx, float ry, sf::Color col) {
    sf::CircleShape c(rx);
    c.setOrigin(rx, rx);
    c.setScale(1.f, ry / rx);
    c.setFillColor(col);
    c.setPosition(x, y);
    rt.draw(c);
}

PortraitContext PortraitGenerator::createContext(unsigned seed, Race race, Gender gender) {
    PortraitContext context{std::mt19937(seed)};
    context.race = race;
    context.gender = gender;

    auto ri = [&](int lo, int hi) {
        return std::uniform_int_distribution<int>(lo, hi)(context.prng);
    };
    auto uc = [](float v) {
        return static_cast<sf::Uint8>(std::clamp(v, 0.f, 255.f));
    };

    // --- COPY YOUR ORIGINAL LOGIC EXACTLY ---

    const sf::Color HUMAN_SKINS[] = {
        {255,219,172}, {240,195,145}, {210,160,110},
        {175,118,80}, {140,90,55}, {100,62,32}
    };
    const sf::Color ORC_SKINS[] = {
        {115,158,78}, {98,140,64}, {82,122,52}, {65,100,42}
    };
    const sf::Color HAIRS[] = {
        {25,15,8}, {75,45,18}, {130,82,32},
        {185,140,60}, {175,55,28}, {190,185,178}
    };
    const sf::Color IRISES[] = {
        {75,48,30}, {50,85,140}, {50,105,65}, {95,98,108}
    };

    int hairMax;

    switch (race) {
        case Race::Elf:   context.skin = HUMAN_SKINS[ri(0,3)]; context.fr=25.f; context.fh=76.f; hairMax=3; break;
        case Race::Dwarf: context.skin = HUMAN_SKINS[ri(0,5)]; context.fr=33.f; context.fh=56.f; hairMax=4; break;
        case Race::Orc:   context.skin = ORC_SKINS[ri(0,3)];   context.fr=28.f; context.fh=66.f; hairMax=4; break;
        default:          context.skin = HUMAN_SKINS[ri(0,5)]; context.fr=28.f; context.fh=68.f; hairMax=4; break;
    }

    context.hair = (race == Race::Orc) ? HAIRS[ri(0,1)] : HAIRS[ri(0,5)];
    context.iris = IRISES[ri(0,3)];

    context.shadow = {uc(context.skin.r*.78f), uc(context.skin.g*.74f), uc(context.skin.b*.71f)};
    context.lip    = {uc(context.skin.r*.70f), uc(context.skin.g*.56f), uc(context.skin.b*.56f)};
    context.brow   = {uc(context.hair.r*.85f+8), uc(context.hair.g*.85f+4), uc(context.hair.b*.82f)};

    // Hair style
    if (gender == Gender::Female) {
        int pool[] = {0,1,1,2,2,4};
        context.hairStyle = pool[ri(0,5)];
        if (context.hairStyle > hairMax) context.hairStyle = hairMax;
    } else {
        context.hairStyle = ri(0, hairMax);
    }

    // Facial hair
    context.fhType = 0;
    if (gender == Gender::Male && race != Race::Elf) {
        if (race == Race::Dwarf) {
            int r2 = ri(0,9);
            if (r2 < 5) context.fhType = 3;
            else if (r2 < 8) context.fhType = 2;
        } else {
            if (ri(0,3) == 0) context.fhType = ri(2,3);
        }
    }

    return context;
}
