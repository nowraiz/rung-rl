import logo from './logo.svg';
import './App.css';
import react from 'react';

class Card extends react.Component {

  render() {
    return (
      <div className="card">
        <img className="card-image" src={"./" + this.props.class + ".png"}></img>
      </div>
    )
  }
}

class PlayedCard extends Card {
  render() {
    if (this.props.empty) {
      return null
    }
    return (
      <div className="card-cell">
        <Card class={this.props.face+this.props.suit} />
      </div>
    )
  }
}

class Hand extends react.Component {
  state = {
    cards: [{ suit: "H", face: "K", hidden: true }, { suit: "S", face: "2" }, { suit: "C", face: "Q"},
    { suit: "H", face: "K", hidden: true },
    { suit: "H", face: "K", hidden: true },
    { suit: "H", face: "K", hidden: true },
    { suit: "H", face: "K", hidden: true },
    { suit: "H", face: "K", hidden: true },
    { suit: "H", face: "K", hidden: true },
    { suit: "H", face: "K", hidden: true },
    { suit: "H", face: "K", hidden: true },
    { suit: "H", face: "K", hidden: true },
    { suit: "H", face: "K", hidden: true },
  ]
  }

  renderCard(suit, face, hidden) {
    if (hidden === true) {
      return <Card class="green_back" />;
    }
    return <Card class={face+suit} />;
  }
  render() {
    return (
      <div className={"hand-"+this.props.id}>
          {
            this.state.cards.map( item => (
              <div className="card-cell">
                {this.renderCard(item.suit, item.face, item.hidden)}
              </div>
            ))
          }
      </div>
    )
  }
}

class Rung extends react.Component {
  
  state = {
    hand_0 : {},
    hand_1 : {},
    hand_2 : {},
    hand_3 : {},
    played_0 : {empty: false, suit: "C", face: "J"},
    played_1 : {empty: false, suit: "C", face: "J"},
    played_2 : {empty: false, suit: "C", face: "J"},
    played_3 : {empty: false, suit: "C", face: "J"},
  }
  render() {

    return (
      <div className="App">
        <Hand id="0" />
        <Hand id="1" />
        <Hand id="2" />
        <Hand id="3" />
        <div className="played-0">
          <PlayedCard empty={this.state.played_0.empty} suit={this.state.played_0.suit} face={this.state.played_0.face} />
        </div>
        <div className="played-1">
          <PlayedCard empty={this.state.played_0.empty} suit={this.state.played_0.suit} face={this.state.played_0.face}/>
        </div>
        <div className="played-2">
        <PlayedCard empty={this.state.played_0.empty} suit={this.state.played_0.suit} face={this.state.played_0.face}/>
        </div>
        <div className="played-3">
        <PlayedCard empty={this.state.played_0.empty} suit={this.state.played_0.suit} face={this.state.played_0.face}/>
        </div>
      </div>
    );
  }
}
function App() {

  return (
    <Rung />
  );
}

export default App;
