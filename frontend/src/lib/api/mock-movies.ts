import type { Movie, Recommendation } from "./movies";

export const MOCK_MOVIES: Movie[] = [
  { id: 1, title: "Inception", year: 2010, short: "A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea." },
  { id: 2, title: "The Dark Knight", year: 2008, short: "Batman raises the stakes in his war on crime with the help of Lt. Jim Gordon and DA Harvey Dent." },
  { id: 3, title: "Interstellar", year: 2014, short: "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival." },
  { id: 4, title: "The Prestige", year: 2006, short: "Two stage magicians engage in a battle to create the ultimate illusion while sacrificing everything." },
  { id: 5, title: "Pulp Fiction", year: 1994, short: "The lives of two mob hitmen, a boxer, and a pair of diner bandits intertwine in four tales of violence." },
  { id: 6, title: "The Matrix", year: 1999, short: "A computer hacker learns about the true nature of reality and his role in the war against its controllers." },
  { id: 7, title: "Fight Club", year: 1999, short: "An insomniac office worker forms an underground fight club that evolves into something much more." },
  { id: 8, title: "Goodfellas", year: 1990, short: "The story of Henry Hill and his life through his years as a mobster with friends Jimmy and Tommy." },
  { id: 9, title: "The Shawshank Redemption", year: 1994, short: "Two imprisoned men bond over years, finding solace and eventual redemption through acts of decency." },
  { id: 10, title: "Parasite", year: 2019, short: "A poor family schemes to become employed by a wealthy family by infiltrating their household." },
  { id: 11, title: "Whiplash", year: 2014, short: "A promising young drummer enrolls at a music conservatory where his dreams are mentored by an instructor." },
  { id: 12, title: "La La Land", year: 2016, short: "A jazz pianist falls for an aspiring actress in Los Angeles while they pursue their dreams." },
  { id: 13, title: "Mad Max: Fury Road", year: 2015, short: "In a post-apocalyptic wasteland, a woman rebels against a tyrannical ruler in search of her homeland." },
  { id: 14, title: "Arrival", year: 2016, short: "A linguist works with the military to communicate with alien lifeforms after twelve spacecraft appear." },
  { id: 15, title: "Blade Runner 2049", year: 2017, short: "A young blade runner's discovery of a long-buried secret leads him to track down former blade runner Rick Deckard." },
  { id: 16, title: "Spirited Away", year: 2001, short: "During her family's move to the suburbs, a sullen 10-year-old wanders into a magical world ruled by a witch." },
  { id: 17, title: "The Grand Budapest Hotel", year: 2014, short: "A writer encounters the owner of an aging high-class hotel, who tells him of his early years as a lobby boy." },
  { id: 18, title: "No Country for Old Men", year: 2007, short: "Violence and mayhem ensue after a hunter stumbles upon a drug deal gone wrong and more than two million dollars." },
  { id: 19, title: "There Will Be Blood", year: 2007, short: "A story of family, religion, hatred, oil and madness, focusing on a turn-of-the-century prospector." },
  { id: 20, title: "Eternal Sunshine of the Spotless Mind", year: 2004, short: "When their relationship turns sour, a couple undergoes a procedure to have each other erased from their memories." },
  { id: 21, title: "Get Out", year: 2017, short: "A young African-American visits his white girlfriend's parents for the weekend, where his unease deepens." },
  { id: 22, title: "Dune", year: 2021, short: "A noble family becomes embroiled in a war for control over the galaxy's most valuable asset." },
  { id: 23, title: "Oppenheimer", year: 2023, short: "The story of American scientist J. Robert Oppenheimer and his role in developing the atomic bomb." },
  { id: 24, title: "Everything Everywhere All at Once", year: 2022, short: "A middle-aged Chinese immigrant is swept up in an adventure where she alone can save existence." },
];

export function mockRecommendations(liked: number[], k = 8): Recommendation[] {
  const pool = MOCK_MOVIES.filter((m) => !liked.includes(m.id));
  const seed = liked.reduce((a, b) => a + b, 0) || 1;
  return pool
    .map((m, i) => ({
      ...m,
      score: Math.max(0.42, Math.min(0.98, 0.95 - ((i * 7 + seed) % 60) / 100)),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}
